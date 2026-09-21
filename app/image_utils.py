"""图片绘制工具：绿色框=保留，红色框=过滤。"""
import base64
import io
from typing import List

import httpx
from PIL import Image, ImageDraw, ImageFont


COLOR_KEEP = (0, 200, 0)
COLOR_FILTER = (220, 30, 30)
COLOR_TEXT = (255, 255, 255)


def _get_font(size=18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def decode_base64_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


async def download_image_from_url(url: str, timeout: float = 20.0) -> Image.Image:
    """从 URL 下载图片并转为 PIL 图像（异步，避免阻塞事件循环）。"""
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")


def analyze_scene(img: Image.Image) -> str:
    """
    分析画面场景特征（夜间/红外黑白/白天强光/孤立亮斑），生成提示词补充约束。
    用于降低"夜晚路灯/补光灯、白天太阳/云/镜头脏污/水珠被误判为火或烟"这类误报。
    返回空串表示无特殊场景。
    """
    gray = img.convert("L")
    hist = gray.histogram()
    total = gray.width * gray.height
    dark_ratio = sum(hist[:64]) / total      # 暗部占比
    bright_ratio = sum(hist[200:]) / total   # 高亮占比

    # 饱和度低 → 黑白/红外夜视画面
    low_sat_ratio = sum(img.convert("HSV").getchannel("S").histogram()[:32]) / total

    # 白天强光/逆光：天空、云、镜头脏污/水珠、太阳眩光是主要干扰源
    is_bright_day = bright_ratio > 0.08 and dark_ratio < 0.3

    hints = []
    if low_sat_ratio > 0.8:
        hints.append("图像为黑白（灰度/红外夜视）画面，无颜色信息")
    if dark_ratio > 0.5:
        hints.append("画面整体昏暗（夜间场景）")
    if bright_ratio > 0.005 and dark_ratio > 0.3:
        hints.append("画面存在局部高亮光斑（典型为夜间灯具/补光灯）")
    if is_bright_day:
        hints.append("画面明亮（白天强光/逆光场景）")

    if not hints:
        return ""

    if is_bright_day:
        suffix = (
            "此类场景下，太阳、眩光、云朵、镜头脏污/水珠形成的高亮或灰白区域均非火焰；"
            "仅当出现清晰的不规则火焰形态并伴随烟雾时才输出 fire。"
            "灰白的云/雾/镜头污渍区域，除非呈现从某点升起、扩散翻滚的烟柱形态，否则不要输出 smoke。"
        )
    else:
        suffix = (
            "此类场景下，孤立的高亮光斑几乎均为灯具、补光灯或反光物，绝非火焰；"
            "除非存在清晰的不规则火焰形态并伴随烟雾，否则禁止输出 fire。"
        )

    return "【场景特征】" + "；".join(hints) + "。" + suffix


def resize_for_llm(img: Image.Image, max_side: int = 1568) -> Image.Image:
    """按最长边缩放到 max_side 以内，用于发送给大模型，控制视觉 token 数量。"""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def encode_image_to_base64(img: Image.Image, fmt="JPEG", quality=90) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def draw_results(image: Image.Image, detections: List[dict]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img, "RGBA")

    w, h = img.size
    base = max(w, h)
    font_size = max(14, int(base / 80))
    line_width = max(2, int(base / 400))
    font = _get_font(font_size)

    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        # 保证坐标合法
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue

        is_real = det.get("is_real", True)
        color = COLOR_KEEP if is_real else COLOR_FILTER

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        class_name = det.get("class_name", "?")
        conf = det.get("confidence", 0.0)
        status = "KEEP" if is_real else "FILTER"
        label = f"{class_name} {conf:.2f} [{status}]"

        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(label) * 8, font_size

        pad = 4
        if y1 - th - pad * 2 > 0:
            bg_y1, bg_y2 = y1 - th - pad * 2, y1
        else:
            bg_y1, bg_y2 = y1, y1 + th + pad * 2

        draw.rectangle([x1, bg_y1, x1 + tw + pad * 2, bg_y2],
                       fill=(*color, 220))
        draw.text((x1 + pad, bg_y1 + pad), label,
                  fill=COLOR_TEXT, font=font)

        reason = det.get("reason", "")
        if reason:
            reason = reason[:40]
            try:
                rb = draw.textbbox((0, 0), reason, font=font)
                rw, rh = rb[2] - rb[0], rb[3] - rb[1]
            except Exception:
                rw, rh = len(reason) * 8, font_size

            ry1 = y2
            ry2 = y2 + rh + pad * 2
            if ry2 > h:
                ry1, ry2 = y2 - rh - pad * 2, y2

            draw.rectangle([x1, ry1, x1 + rw + pad * 2, ry2],
                           fill=(0, 0, 0, 180))
            draw.text((x1 + pad, ry1 + pad), reason,
                      fill=(255, 255, 255), font=font)

    return img