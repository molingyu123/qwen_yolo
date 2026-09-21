"""路由定义。"""
import os
import time
import uuid
from fastapi import APIRouter, HTTPException
from app.config import settings
from app.logger import log
from app.schemas import (
    FilterRequest, FilterResponse, FilterResult,
    FilterWithImageRequest, FilterWithImageResponse,
    DetectRequest, DetectResponse, DetectItem,
    HealthResponse,
)
from app.prompts import validate_custom_prompt
from app.qwen_client import qwen_client
from app.image_utils import (
    decode_base64_image, download_image_from_url, encode_image_to_base64,
    resize_for_llm, draw_results, analyze_scene,
)

router = APIRouter()


def _validate_prompt_or_raise(custom_prompt: str, request_id: str):
    """校验自定义提示词，不合法直接抛 400。"""
    if not custom_prompt:
        return
    ok, err = validate_custom_prompt(custom_prompt)
    if not ok:
        log.warning(f"[{request_id}] 提示词校验失败: {err}")
        raise HTTPException(
            status_code=400,
            detail=f"custom_prompt 不合法: {err}",
        )


def _handle_detection(det) -> tuple:
    """根据置信度分流。返回 (action, is_real, reason)。"""
    if det.confidence < settings.confidence_low:
        return "filter", False, f"置信度 {det.confidence:.2f} 低于阈值"
    if det.confidence > settings.confidence_high:
        return "keep", True, f"置信度 {det.confidence:.2f} 高于阈值"
    return None, None, None  # 需要大模型复核


async def _process_detections(req, request_id: str):
    """处理所有检测项，返回 results。"""
    # 场景分析基于原图（而非各检测项的裁剪图）算一次即可：裁剪图本身缺乏
    # 全局上下文（是否夜间/逆光/红外画面），靠原图的整体明暗/饱和度分布补上。
    scene_hint = ""
    try:
        original = decode_base64_image(req.image_base64)
        scene_hint = analyze_scene(original)
    except Exception as e:
        log.warning(f"[{request_id}] 场景分析失败，跳过: {e}")

    camera_type = getattr(req, "camera_type", "visible")

    results = []
    for det in req.detections:
        action, is_real, reason = _handle_detection(det)

        if action is None:
            is_real, reason = await qwen_client.filter_detection(
                image_base64=req.image_base64,
                class_name=det.class_name,
                confidence=det.confidence,
                custom_prompt=req.custom_prompt,
                camera_type=camera_type,
                scene_hint=scene_hint,
            )
            action = "keep" if is_real else "filter"

        results.append(FilterResult(
            detection_id=det.detection_id,
            class_name=det.class_name,
            is_real=is_real,
            reason=reason,
            action=action,
        ))
    return results


@router.get("/health", response_model=HealthResponse, tags=["系统"])
async def health():
    """健康检查，同时检查 vLLM 后端。"""
    qwen_ok = await qwen_client.health_check()
    return HealthResponse(
        status="ok",
        qwen_service="ok" if qwen_ok else "unavailable",
        version=settings.app_version,
    )


@router.post("/filter", response_model=FilterResponse, tags=["核心"])
async def filter_detections(req: FilterRequest):
    """二次过滤，不返回图片。"""
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    log.info(f"[{request_id}] 过滤请求, detections={len(req.detections)}")

    # 校验自定义提示词
    _validate_prompt_or_raise(req.custom_prompt, request_id)

    results = await _process_detections(req, request_id)

    elapsed = (time.perf_counter() - start) * 1000
    log.info(f"[{request_id}] 完成, 耗时 {elapsed:.1f}ms")

    return FilterResponse(
        request_id=request_id,
        results=results,
        elapsed_ms=round(elapsed, 2),
    )


@router.post(
    "/filter-with-image",
    response_model=FilterWithImageResponse,
    tags=["核心"],
)
async def filter_with_image(req: FilterWithImageRequest):
    """二次过滤，返回标注后的图片。"""
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    log.info(f"[{request_id}] 带图过滤, detections={len(req.detections)}")

    # 校验自定义提示词
    _validate_prompt_or_raise(req.custom_prompt, request_id)

    results = await _process_detections(req, request_id)

    # 绘制标注图
    try:
        original = decode_base64_image(req.image_base64)
        draw_items = []
        for r, d in zip(results, req.detections):
            draw_items.append({
                "bbox": d.bbox or [],
                "class_name": r.class_name,
                "confidence": d.confidence,
                "is_real": r.is_real,
                "reason": r.reason,
            })
        annotated = draw_results(original, draw_items)
        annotated_b64 = encode_image_to_base64(
            annotated, fmt=req.image_format,
        )
    except Exception as e:
        log.exception(f"[{request_id}] 绘制标注图失败: {e}")
        annotated_b64 = req.image_base64  # 兜底返回原图

    elapsed = (time.perf_counter() - start) * 1000
    log.info(f"[{request_id}] 完成, 耗时 {elapsed:.1f}ms")

    return FilterWithImageResponse(
        request_id=request_id,
        results=results,
        annotated_image_base64=annotated_b64,
        elapsed_ms=round(elapsed, 2),
    )


def _resolve_categories(req) -> list[str]:
    """解析请求的识别类别，未指定则用默认配置。"""
    if req.categories:
        cats = [c.strip() for c in req.categories if c and c.strip()]
        if cats:
            return cats
    return [c.strip() for c in settings.detect_categories.split(",") if c.strip()]


def _save_annotated(img, format_ext: str) -> tuple[str, str]:
    """保存标注图，返回 (相对文件名, 磁盘路径)，并确保目录存在。"""
    os.makedirs(settings.annotation_output_dir, exist_ok=True)
    fname = f"{uuid.uuid4().hex[:16]}.{format_ext.lower()}"
    path = os.path.join(settings.annotation_output_dir, fname)
    img.save(path, format=format_ext)
    return fname, path


@router.post("/detect", response_model=DetectResponse, tags=["核心"])
async def detect_image(req: DetectRequest):
    """整图识别：传图片地址，大模型自动检测预设类别目标，返回标注图。"""
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    log.info(f"[{request_id}] 整图识别请求, url={bool(req.image_url)}")

    # 校验自定义提示词
    _validate_prompt_or_raise(req.custom_prompt, request_id)

    # 1. 获取原图
    try:
        if req.image_url:
            original = await download_image_from_url(req.image_url)
        else:
            original = decode_base64_image(req.image_base64)  # type: ignore[arg-type]
    except Exception as e:
        log.warning(f"[{request_id}] 获取图片失败: {e}")
        raise HTTPException(status_code=400, detail=f"图片获取失败: {e}")

    # 2. 调用大模型识别（先缩放，控制视觉 token，避免超上下文）
    categories = _resolve_categories(req)
    img_for_llm = encode_image_to_base64(resize_for_llm(original), fmt="JPEG")
    # 场景感知：夜间/红外黑白画面自动注入补充约束，压"灯光误判为火"误报
    scene_hint = analyze_scene(original)
    if scene_hint:
        log.info(f"[{request_id}] 场景提示: {scene_hint[:60]}...")
    detections, reason, recognized = await qwen_client.detect_image(
        image_base64=img_for_llm,
        categories=categories,
        custom_prompt=req.custom_prompt,
        scene_hint=scene_hint,
        camera_type=getattr(req, "camera_type", "visible"),
    )
    log.info(f"[{request_id}] 识别到 {len(detections)} 个目标")

    # bbox 由 YOLO 归一化(0~1) 转为像素坐标，供绘制与响应使用
    img_w, img_h = original.size
    for d in detections:
        if len(d["bbox"]) == 4:
            x1, y1, x2, y2 = d["bbox"]
            d["bbox"] = [x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h]

    # 3. 绘制标注图：检出结果含 fire/smoke（火情/烟雾）才画框；
    #    否则即使检出 person/vehicle 等其他类别，也不画框，直接返回原图
    fire_smoke_set = {
        c.strip().lower()
        for c in settings.fire_smoke_categories.split(",")
        if c.strip()
    }
    has_fire_smoke = any(
        d["class_name"].lower() in fire_smoke_set for d in detections
    )
    draw_items = []
    if has_fire_smoke:
        for d in detections:
            draw_items.append({
                "bbox": d["bbox"],
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "is_real": True,
                "reason": "",
            })
    annotated = draw_results(original, draw_items)

    # 4. 保存标注图并生成可访问 URL（调用方自行下载，不返回 base64）
    save_fmt = "PNG" if req.image_format == "PNG" else "JPEG"
    fname, _ = _save_annotated(annotated, save_fmt)
    annotated_url = f"{settings.annotated_base_url.rstrip('/')}/annotated/{fname}"

    elapsed = (time.perf_counter() - start) * 1000
    log.info(f"[{request_id}] 完成, 耗时 {elapsed:.1f}ms")

    return DetectResponse(
        request_id=request_id,
        detections=[
            DetectItem(
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=d["bbox"] or None,
            )
            for d in detections
        ],
        annotated_image_url=annotated_url,
        recognized=recognized,
        has_detection=len(detections) > 0,
        reason=reason,
        elapsed_ms=round(elapsed, 2),
    )