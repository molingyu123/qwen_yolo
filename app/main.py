"""FastAPI 入口，含安全中间件。"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.logger import log
from app.qwen_client import qwen_client
from app.routes import router


class SecurityMiddleware(BaseHTTPMiddleware):
    """IP 白名单校验（API Key 校验已移除）。"""

    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 豁免路径
        if path in self.EXEMPT_PATHS or path.endswith("/health"):
            return await call_next(request)

        # 标注图静态目录无需鉴权（供 img 直接访问）
        if path.startswith("/annotated/"):
            return await call_next(request)

        # IP 白名单
        if settings.allow_ips:
            allowed = {
                ip.strip()
                for ip in settings.allow_ips.split(",")
                if ip.strip()
            }
            client_ip = request.client.host if request.client else ""
            xff = request.headers.get("X-Forwarded-For", "")
            real_ip = xff.split(",")[0].strip() if xff else client_ip

            if real_ip not in allowed and real_ip != "127.0.0.1":
                log.warning(f"拒绝非白名单 IP: {real_ip} 访问 {path}")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "code": "FORBIDDEN",
                        "message": "IP 不在白名单",
                    },
                )

        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        f"===== {settings.app_name} v{settings.app_version} 启动 ====="
    )
    log.info(f"监听地址: {settings.host}:{settings.port}")
    log.info("鉴权: API Key 校验已关闭")
    log.info(f"IP 白名单: {settings.allow_ips}")
    # 确保标注图输出目录存在
    os.makedirs(settings.annotation_output_dir, exist_ok=True)
    await qwen_client.startup()
    yield
    log.info("===== 应用关闭 =====")
    await qwen_client.shutdown()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(SecurityMiddleware)


@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    """兜底异常处理，不泄漏堆栈。"""
    log.exception(f"未捕获异常: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": "INTERNAL_ERROR",
            "message": "服务器内部错误",
        },
    )


app.include_router(router, prefix="/api/v1")

# 挂载标注图静态目录，供 annotated_image_url 直接访问
os.makedirs(settings.annotation_output_dir, exist_ok=True)
app.mount(
    "/annotated",
    StaticFiles(directory=settings.annotation_output_dir),
    name="annotated",
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )