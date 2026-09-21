"""请求 / 响应模型，带输入校验。"""
import base64
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


def _clean_base64(v: str) -> str:
    """去掉 data URI 前缀并校验 base64。"""
    if "," in v:
        v = v.split(",", 1)[1]
    try:
        base64.b64decode(v, validate=True)
    except Exception:
        raise ValueError("image_base64 不是合法的 base64")
    return v


class DetectionItem(BaseModel):
    detection_id: str = Field(..., min_length=1, max_length=64)
    class_name: str = Field(..., min_length=1, max_length=64)
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Optional[List[float]] = Field(default=None, max_length=4)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("bbox 必须为 [x1, y1, x2, y2]")
        return v


class FilterRequest(BaseModel):
    image_base64: str = Field(..., min_length=10, max_length=20_000_000)
    detections: List[DetectionItem] = Field(..., min_length=1, max_length=20)
    custom_prompt: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("image_base64")
    @classmethod
    def validate_b64(cls, v):
        return _clean_base64(v)


class FilterResult(BaseModel):
    detection_id: str
    class_name: str
    is_real: bool
    reason: str = ""
    action: str


class FilterResponse(BaseModel):
    request_id: str
    results: List[FilterResult]
    elapsed_ms: float


class FilterWithImageRequest(BaseModel):
    image_base64: str = Field(..., min_length=10, max_length=50_000_000)
    detections: List[DetectionItem] = Field(..., min_length=1, max_length=50)
    custom_prompt: Optional[str] = Field(default=None, max_length=1000)
    image_format: str = Field(default="JPEG", pattern="^(JPEG|PNG)$")

    @field_validator("image_base64")
    @classmethod
    def validate_b64(cls, v):
        return _clean_base64(v)


class FilterWithImageResponse(BaseModel):
    request_id: str
    results: List[FilterResult]
    annotated_image_base64: str
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    qwen_service: str
    version: str


class DetectRequest(BaseModel):
    image_url: Optional[str] = Field(default=None, max_length=4096)
    # 兼容方案：也可直接传 base64（二者必填其一，优先 image_url）
    image_base64: Optional[str] = Field(default=None, min_length=10, max_length=50_000_000)
    categories: Optional[List[str]] = Field(default=None, max_length=50)
    custom_prompt: Optional[str] = Field(default=None, max_length=1000)
    image_format: str = Field(default="JPEG", pattern="^(JPEG|PNG)$")

    @field_validator("image_base64")
    @classmethod
    def validate_b64(cls, v):
        if v is None:
            return v
        return _clean_base64(v)

    @model_validator(mode="after")
    def check_image_source(self):
        if bool(self.image_url) == bool(self.image_base64):
            raise ValueError("image_url 与 image_base64 必须且只能提供一个")
        return self


class DetectItem(BaseModel):
    class_name: str = Field(..., min_length=1, max_length=64)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: Optional[List[float]] = Field(default=None, max_length=4)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("bbox 必须为 [x1, y1, x2, y2]")
        return v


class DetectResponse(BaseModel):
    request_id: str
    detections: List[DetectItem]
    annotated_image_url: str
    recognized: bool = Field(..., description="识别服务是否正常执行完成（与是否检出目标无关）")
    has_detection: bool = Field(..., description="是否检测到目标（detections 非空即为 true）")
    reason: str = ""
    elapsed_ms: float