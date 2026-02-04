"""
Anima LoRA Trainer GUI - Backend
================================
FastAPI 后端主应用
参考 akegarasu/lora-scripts 架构设计

特性:
- 单端口架构 (28000)
- 静态文件挂载 (前端编译产物)
- RESTful API
- 训练进程管理
"""

import os
import sys
import mimetypes
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException

# 确保可以导入项目根目录的模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gui.backend.api import config_router, train_router, system_router
from gui.backend.api.train import list_checkpoints
from gui.backend.services.trainer import trainer_service
from gui.backend.log import log

# 修复 MIME 类型
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


class SPAStaticFiles(StaticFiles):
    """SPA 静态文件处理器 - 所有路由返回 index.html"""
    
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as ex:
            if ex.status_code == 404:
                # 对于 404 错误，返回 index.html (SPA 前端处理路由)
                return await super().get_response("index.html", scope)
            else:
                raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    log.info("[START] Anima LoRA Trainer GUI backend starting...")
    log.info(f"[PATH] Working directory: {project_root}")
    
    # 确保必要的目录存在
    os.makedirs(project_root / "output", exist_ok=True)
    os.makedirs(project_root / "config" / "autosave", exist_ok=True)
    
    yield
    
    # 关闭时
    log.info("[STOP] Shutting down...")
    trainer_service.stop_all()


# 创建 FastAPI 应用
app = FastAPI(
    title="Anima LoRA Trainer GUI",
    description="Anima 模型 LoRA 训练图形界面",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置 (开发模式启用)
if os.environ.get("ANIMA_GUI_DEV", "0") == "1":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 注册 API 路由
app.include_router(config_router, prefix="/api/config", tags=["配置管理"])
app.include_router(train_router, prefix="/api/train", tags=["训练控制"])
app.include_router(system_router, prefix="/api/system", tags=["系统信息"])


@app.get("/api/checkpoint/list")
async def checkpoint_list_legacy():
    """兼容旧前端的 checkpoint 列表接口"""
    return await list_checkpoints()


def _resolve_checkpoint_path(path: str) -> Path:
    output_dir = project_root / "output"
    target = Path(path).resolve()
    if not target.exists():
        raise HTTPException(status_code=404, detail="路径不存在")
    if output_dir not in target.parents and target != output_dir:
        raise HTTPException(status_code=400, detail="非法路径")
    return target


def _resolve_output_filename(filename: str) -> Path:
    output_dir = project_root / "output"
    if not filename:
        raise HTTPException(status_code=400, detail="文件名无效")
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="文件名非法")
    target = (output_dir / safe_name).resolve()
    if output_dir not in target.parents and target != output_dir:
        raise HTTPException(status_code=400, detail="非法路径")
    return target


@app.delete("/api/checkpoint")
async def checkpoint_delete(path: str):
    """兼容旧前端的 checkpoint 删除接口"""
    target = _resolve_checkpoint_path(path)
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"status": "success"}


@app.post("/api/checkpoint/rename")
async def checkpoint_rename(payload: dict):
    """兼容旧前端的 checkpoint 重命名接口"""
    old_path = payload.get("oldPath")
    new_path = payload.get("newPath")
    if not old_path or not new_path:
        raise HTTPException(status_code=400, detail="缺少路径参数")

    target = _resolve_checkpoint_path(old_path)
    new_target = Path(new_path)
    if not new_target.is_absolute():
        new_target = target.parent / new_target
    new_target = new_target.resolve()
    output_dir = project_root / "output"
    if output_dir not in new_target.parents and new_target != output_dir:
        raise HTTPException(status_code=400, detail="非法路径")
    if new_target.exists():
        raise HTTPException(status_code=409, detail="目标路径已存在")
    target.rename(new_target)
    return {"status": "success"}


@app.post("/api/checkpoint/upload")
async def checkpoint_upload(request: Request):
    """兼容旧前端的 checkpoint 上传接口"""
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except RuntimeError:
            raise HTTPException(
                status_code=400,
                detail="需要安装 python-multipart 才能上传文件",
            )
        file_obj = form.get("file")
        file_handle = getattr(file_obj, "file", None)
        if file_obj is None or file_handle is None:
            raise HTTPException(status_code=400, detail="未找到上传文件")
        filename = getattr(file_obj, "filename", None)
        target = _resolve_output_filename(filename or "")
        if target.exists():
            raise HTTPException(status_code=409, detail="文件已存在")
        with target.open("wb") as f:
            shutil.copyfileobj(file_handle, f)
        return {"status": "success", "path": str(target)}

    filename = request.query_params.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="缺少 filename 参数")
    target = _resolve_output_filename(filename)
    if target.exists():
        raise HTTPException(status_code=409, detail="文件已存在")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="文件内容为空")
    with target.open("wb") as f:
        f.write(body)
    return {"status": "success", "path": str(target)}


@app.get("/api/checkpoint/download")
async def checkpoint_download(path: str):
    """兼容旧前端的 checkpoint 下载接口"""
    target = _resolve_checkpoint_path(path)
    if target.is_dir():
        raise HTTPException(status_code=400, detail="不支持下载目录")
    return FileResponse(str(target))


@app.get("/")
async def index():
    """根路径返回前端入口"""
    frontend_path = Path(__file__).parent.parent / "frontend" / "dist" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return {"message": "Anima LoRA Trainer GUI - 前端未编译"}


# 挂载前端静态文件
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", SPAStaticFiles(directory=str(frontend_dist), html=True), name="static")
else:
    log.warning(f"前端目录不存在: {frontend_dist}，请先编译前端")
