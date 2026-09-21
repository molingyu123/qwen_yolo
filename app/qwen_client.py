"""vLLM 客户端封装：超时、重试、保守失败、JSON 容错解析。"""
import asyncio
import json
import re
from typing import Optional, Tuple

import httpx
from app.config import settings
from app.logger import log
from app.prompts import (
    build_prompt, build_detection_prompt, validate_custom_prompt,
)


class QwenClient:
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def startup(self):
        self._client = httpx.AsyncClient(
            base_url=settings.qwen_base_url,
            timeout=httpx.Timeout(
                connect=5.0,
                read=settings.qwen_timeout,
                write=5.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=64,
                max_keepalive_connections=16,
            ),
            headers={
                "Authorization": f"Bearer {settings.qwen_api_key}",
                "Content-Type": "application/json",
            },
        )
        log.info(f"QwenClient 初始化: {settings.qwen_base_url}")

    async def shutdown(self):
        if self._client:
            await self._client.aclose()
            log.info("QwenClient 已关闭")

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/models", timeout=5.0)
            return resp.status_code == 200
        except Exception as e:
            log.warning(f"vLLM 健康检查失败: {e}")
            return False

    async def filter_detection(
        self,
        image_base64: str,
        class_name: str,
        confidence: float,
        custom_prompt: Optional[str] = None,
        camera_type: str = "visible",
        scene_hint: str = "",
    ) -> Tuple[bool, str]:
        """复核单个检测项，返回 (is_real, reason)。

        camera_type: "visible"（可见光，默认）或 "thermal"（热成像），
            决定使用哪一套误报知识库（颜色线索对热成像完全失效）。
        scene_hint: 对原图（而非裁剪图）做场景分析后的补充约束，
            用于弥补裁剪图缺失全局上下文（夜间/逆光等）的问题。
        """
        # ===== 安全校验：自定义提示词 =====
        if custom_prompt:
            ok, err = validate_custom_prompt(custom_prompt)
            if not ok:
                log.warning(f"自定义提示词被拒绝: {err}")
                custom_prompt = None  # 回退到默认提示词

        # ===== 构造提示词 =====
        prompt = build_prompt(
            class_name, confidence, custom_prompt,
            camera_type=camera_type, scene_hint=scene_hint,
        )

        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        payload = {
            "model": settings.qwen_model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": settings.qwen_max_tokens,
            "temperature": settings.qwen_temperature,
            "top_p": settings.qwen_top_p,
            "frequency_penalty": settings.qwen_frequency_penalty,
        }

        last_err: Optional[Exception] = None
        for attempt in range(settings.qwen_max_retries + 1):
            try:
                resp = await self._client.post(
                    "/chat/completions", json=payload,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return self._parse_and_validate(content, class_name)

            except httpx.TimeoutException as e:
                last_err = e
                log.warning(f"vLLM 超时 (第 {attempt + 1} 次)")
            except httpx.HTTPStatusError as e:
                last_err = e
                log.error(f"vLLM 状态 {e.response.status_code}: "
                          f"{e.response.text[:200]}")
                if 400 <= e.response.status_code < 500:
                    break
            except Exception as e:
                last_err = e
                log.error(f"vLLM 调用异常 (第 {attempt + 1} 次): {e}")

            if attempt < settings.qwen_max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))

        log.error(f"vLLM 调用最终失败，保守保留: {last_err}")
        return True, "复核服务异常，保守保留"

    async def detect_image(
        self,
        image_base64: str,
        categories: list[str],
        custom_prompt: Optional[str] = None,
        scene_hint: str = "",
        camera_type: str = "visible",
    ) -> tuple[list[dict], str, bool]:
        """
        整图识别：让大模型从全图中检测指定类别的目标。
        返回 (detections, reason, recognized)。
        - detections: dict 列表 {"class_name","confidence","bbox"}
        - reason: 失败原因；成功时为空串
        - recognized: True=模型正常返回；False=调用失败（与是否检出目标无关）
        - scene_hint: 场景补充约束（如夜间/红外画面提示）
        - camera_type: "visible" 或 "thermal"，决定是否附加热成像专用规则
        """
        # ===== 安全校验：自定义提示词 =====
        if custom_prompt:
            ok, err = validate_custom_prompt(custom_prompt)
            if not ok:
                log.warning(f"自定义提示词被拒绝: {err}")
                custom_prompt = None  # 回退到检测模板

        # ===== 构造提示词 =====
        prompt = build_detection_prompt(
            categories, custom_prompt, scene_hint, camera_type=camera_type,
        )

        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        payload = {
            "model": settings.qwen_model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": settings.qwen_detect_max_tokens,
            "temperature": settings.qwen_temperature,
            "top_p": settings.qwen_top_p,
        }

        last_err: Optional[Exception] = None
        abort_reason: Optional[str] = None
        for attempt in range(settings.qwen_max_retries + 1):
            try:
                resp = await self._client.post(
                    "/chat/completions", json=payload,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                dets, r = self._parse_detections(content, categories)
                return dets, r, True

            except httpx.TimeoutException as e:
                last_err = e
                log.warning(f"vLLM 识别超时 (第 {attempt + 1} 次)")
            except httpx.HTTPStatusError as e:
                last_err = e
                body = e.response.text[:500]
                log.error(f"vLLM 状态 {e.response.status_code}: {body}")
                if 400 <= e.response.status_code < 500:
                    abort_reason = f"模型服务拒绝请求: HTTP {e.response.status_code} {body}"
                    break
            except Exception as e:
                last_err = e
                log.error(f"vLLM 识别异常 (第 {attempt + 1} 次): {e}")

            if attempt < settings.qwen_max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))

        log.error(f"vLLM 识别最终失败: {abort_reason or last_err}")
        return [], abort_reason or "识别服务异常", False

    @staticmethod
    def _parse_detections(
        content: str, categories: list[str],
    ) -> tuple[list[dict], str]:
        """
        解析并校验整图识别结果（has_find + objects_analysis + YOLO 归一化 bbox）。
        返回的 bbox 为归一化坐标 [x1, y1, x2, y2]（0~1），由调用方换算成像素坐标。
        置信度 < 0.7 的目标直接丢弃（与提示词中过滤规则一致）。
        """
        if not content:
            return [], "模型返回空内容"

        text = content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            log.warning(f"无法从响应提取 JSON 对象: {content[:200]}")
            return [], "响应解析失败，未检测到目标"

        try:
            data = json.loads(m.group())
        except json.JSONDecodeError as e:
            log.warning(f"JSON 解析失败: {e}")
            return [], "JSON 解析失败，未检测到目标"

        if not isinstance(data, dict):
            return [], "响应格式异常，未检测到目标"

        log.info(f"模型返回 JSON: {json.dumps(data, ensure_ascii=False)[:800]}")

        # has_find 支持两种格式：{"value": true} 或直接 true
        has_find_raw = data.get("has_find")
        if isinstance(has_find_raw, dict):
            has_find = has_find_raw.get("value", True)
        else:
            has_find = has_find_raw if isinstance(has_find_raw, bool) else True
        raw_list = data.get("objects_analysis") or []
        if not has_find or not isinstance(raw_list, list):
            return [], ""

        # YOLO class_id → 类别名：0=smoke，1=fire
        CLASS_BY_ID = {"0": "smoke", "1": "fire"}
        detections = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            log.info(f"解析检测项: {item}")

            # 类别：优先 name / label_code（smoke|fire），否则按 class_id 兜底
            name = str(item.get("name") or "").strip().lower()
            label = str(item.get("label_code") or "").strip().lower()
            class_name = (
                name if name in ("smoke", "fire")
                else (label if label in ("smoke", "fire") else "")
            )

            # 置信度：支持 {"value": 0.95} 或直接 0.95
            conf_obj = item.get("confidence")
            if isinstance(conf_obj, dict):
                confidence = conf_obj.get("value")
            else:
                confidence = conf_obj
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            if confidence < 0.7:
                continue

            # YOLO 归一化 bbox 解析，兼容多种格式
            bbox = QwenClient._parse_yolo_bbox(item)
            if not bbox and not class_name:
                # 尝试从 bbox class_id 推断类别
                raw_bbox = item.get("yolo_bbox")
                if isinstance(raw_bbox, dict):
                    raw_bbox = raw_bbox.get("value", "")
                elif isinstance(raw_bbox, (list, tuple)):
                    raw_bbox = " ".join(str(v) for v in raw_bbox)
                parts = str(raw_bbox or "").split()
                if len(parts) >= 1:
                    try:
                        cid = float(parts[0])
                        class_name = CLASS_BY_ID.get(str(int(cid)), "")
                    except (TypeError, ValueError):
                        pass

            if not class_name:
                continue

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox,
            })

        if not detections and raw_list:
            log.warning(f"解析后无有效检测项, 原始: {content[:500]}")
        for d in detections:
            log.info(f"解析结果: class={d['class_name']} conf={d['confidence']} bbox={d['bbox']}")
        return detections, ""

    @staticmethod
    def _parse_yolo_bbox(item: dict) -> list[float]:
        """
        从检测项解析 YOLO 归一化 bbox，返回 [x1, y1, x2, y2]（0~1）。
        兼容多种格式：
        - {"yolo_bbox": {"value": "cid cx cy w h"}}
        - {"yolo_bbox": "cid cx cy w h"}
        - {"bbox": [x1, y1, x2, y2]}（已是 xyxy）
        - {"bbox": [cx, cy, w, h]}（需判断是 xywh 还是 xyxy）
        - {"x_center","y_center","width","height"} 分离字段
        """
        # 格式1: yolo_bbox.value 字符串，灵活解析
        raw = item.get("yolo_bbox")
        if isinstance(raw, dict):
            raw = raw.get("value", "")
        if isinstance(raw, (list, tuple)):
            raw = " ".join(str(v) for v in raw)
        if isinstance(raw, str):
            parts = raw.strip().split()
            try:
                vals = [float(p) for p in parts]
            except (TypeError, ValueError):
                vals = []
            # 5值+ → 跳过 class_id，取后4个；4值 → 直接用；3值 → 补0宽高
            if len(vals) >= 5:
                vals = vals[1:5]
            elif len(vals) == 4:
                pass  # cx cy w h
            elif len(vals) == 3:
                vals.append(0)  # 缺 height 补 0
                vals.insert(2, 0)  # 缺 width 补 0 → cx cy 0 0
            else:
                vals = []
            if len(vals) == 4:
                cx, cy, bw, bh = vals
                return [cx - bw / 2, cy - bh / 2,
                        cx + bw / 2, cy + bh / 2]

        # 格式2: bbox = [x1, y1, x2, y2] 或 [cx, cy, w, h]
        bbox_raw = item.get("bbox")
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            try:
                vals = [float(v) for v in bbox_raw]
                # 如果值都在 0~1 范围，假设是归一化坐标
                if all(0.0 <= v <= 1.0 for v in vals):
                    # 判断是 xyxy 还是 xywh：xywh 时 w+h 通常 < 2，xyxy 时 x2>x1 且 y2>y1
                    if vals[2] < vals[0] or vals[3] < vals[1]:
                        # w/h 可能比 cx/cy 小（xywh 格式）
                        cx, cy, bw, bh = vals
                        return [cx - bw / 2, cy - bh / 2,
                                cx + bw / 2, cy + bh / 2]
                    return vals  # 已经是 [x1, y1, x2, y2]
                else:
                    # 像素坐标，直接返回（由调用方处理）
                    return vals
            except (TypeError, ValueError):
                pass

        # 格式3: 分离字段 x_center/y_center/width/height
        try:
            cx = float(item.get("x_center", item.get("cx", 0)))
            cy = float(item.get("y_center", item.get("cy", 0)))
            bw = float(item.get("width", item.get("w", 0)))
            bh = float(item.get("height", item.get("h", 0)))
            if bw > 0 and bh > 0:
                return [cx - bw / 2, cy - bh / 2,
                        cx + bw / 2, cy + bh / 2]
        except (TypeError, ValueError):
            pass

        return []

    @staticmethod
    def _parse_and_validate(
        content: str, class_name: str,
    ) -> Tuple[bool, str]:
        """解析并校验模型输出。"""
        if not content:
            return True, "模型返回空内容，保守保留"

        text = content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if not m:
            log.warning(f"无法从响应提取 JSON: {content[:200]}")
            return True, "响应解析失败，保守保留"

        try:
            data = json.loads(m.group())
        except json.JSONDecodeError as e:
            log.warning(f"JSON 解析失败: {e}")
            return True, "JSON 解析失败，保守保留"

        # ===== 字段校验 =====
        if "is_real" not in data:
            return True, "缺少 is_real 字段，保守保留"

        is_real = data.get("is_real")
        if not isinstance(is_real, bool):
            return True, "is_real 类型异常，保守保留"

        reason = str(data.get("reason", ""))[:200]
        # 去掉控制字符
        reason = re.sub(r"[\x00-\x1f\x7f]", "", reason)

        return is_real, reason


qwen_client = QwenClient()