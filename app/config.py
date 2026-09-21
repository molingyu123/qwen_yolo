"""配置管理。所有参数硬编码于此，不读取 .env / .env.example。"""
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # ===== 服务 =====
    app_name: str = Field(default="qwen-filter-service")
    app_version: str = Field(default="1.0.0")
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=9070, ge=1, le=65535)
    debug: bool = Field(default=False)

    # ===== 访问控制 =====
    # IP 白名单，逗号分隔；设为空串表示不限制
    allow_ips: str = Field(default="127.0.0.1,::1")

    # ===== vLLM =====
    qwen_base_url: str = Field(default="http://127.0.0.1:9080/v1")
    qwen_model_name: str = Field(default="qwen2.5-vl-7b")
    qwen_api_key: str = Field(default="EMPTY")
    qwen_timeout: float = Field(default=30.0, gt=0, le=120)
    qwen_max_retries: int = Field(default=2, ge=0, le=5)
    qwen_max_tokens: int = Field(default=128, ge=1, le=2048)
    qwen_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    qwen_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    qwen_frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    # 整图检测时的最大输出 token（需要容纳多个目标）
    qwen_detect_max_tokens: int = Field(default=1024, ge=256, le=4096)

    # ===== 业务 =====
    confidence_low: float = Field(default=0.3, ge=0.0, le=1.0)
    confidence_high: float = Field(default=0.7, ge=0.0, le=1.0)

    # 自定义提示词最大长度
    max_custom_prompt_len: int = Field(default=600, ge=100, le=2000)

    # ===== 整图识别 =====
    # 预设识别类别，逗号分隔（对应 prompts.py 的 CATEGORY_GUIDES 键）
    detect_categories: str = Field(
        default="fire,smoke,person,vehicle,helmet,vest",
    )
    # 只有检出这些类别（火情/烟雾）时才在标注图上画框；其余类别检出也不画框
    fire_smoke_categories: str = Field(default="fire,smoke")

    # ===== 标注图输出 =====
    annotation_output_dir: str = Field(default="./annotated")
    # 对外可访问的标注图 URL 前缀
    annotated_base_url: str = Field(default="http://127.0.0.1:9070")

    # ===== 日志 =====
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="./logs")


settings = Settings()