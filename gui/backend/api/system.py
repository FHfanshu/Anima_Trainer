"""
系统信息 API
提供 GPU 信息、文件选择等功能
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gui.backend.log import log

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class SystemResponse(BaseModel):
    status: str
    message: str = ""
    data: Optional[Dict[str, Any]] = None


@router.get("/info", response_model=SystemResponse)
async def get_system_info():
    """获取系统信息"""
    try:
        info: Dict[str, Any] = {}
        info["platform"] = platform.system()
        info["python_version"] = platform.python_version()
        info["python_executable"] = sys.executable
        info["working_directory"] = str(PROJECT_ROOT)
        
        # 尝试获取 GPU 信息
        gpu_info = []
        try:
            import torch  # type: ignore
            info["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_info.append({
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory": torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
                    })
            info["gpus"] = gpu_info
            info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        except ImportError:
            info["gpus"] = []
            info["cuda_version"] = None
            info["torch_version"] = None

        try:
            import diffusers  # type: ignore
            info["diffusers_version"] = diffusers.__version__
        except ImportError:
            info["diffusers_version"] = None
        
        return SystemResponse(status="success", data=info)
        
    except Exception as e:
        log.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gpu_status", response_model=SystemResponse)
async def get_gpu_status():
    """获取 GPU 实时状态"""
    try:
        gpus = []
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
                errors="replace",
            )
            for line in output.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                idx, name, total_mb, used_mb = parts[:4]
                try:
                    total = float(total_mb) / 1024
                    used = float(used_mb) / 1024
                except ValueError:
                    continue
                utilization = round(used / total * 100, 1) if total > 0 else 0
                gpus.append({
                    "id": int(idx) if idx.isdigit() else idx,
                    "name": name,
                    "allocated_gb": round(used, 2),
                    "reserved_gb": round(used, 2),
                    "total_gb": round(total, 2),
                    "utilization": utilization,
                })
            return SystemResponse(status="success", data={"gpus": gpus})
        except Exception:
            pass

        import torch  # type: ignore
        if not torch.cuda.is_available():
            return SystemResponse(status="success", data={"gpus": []})

        gpus = []
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)

            gpus.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "utilization": round(allocated / total * 100, 1) if total > 0 else 0,
            })

        return SystemResponse(status="success", data={"gpus": gpus})

    except ImportError:
        return SystemResponse(status="success", data={"gpus": []})


@router.get("/gpu")
async def get_gpu_legacy():
    """兼容旧前端的 GPU 接口"""
    response = await get_gpu_status()
    gpus = response.data.get("gpus", []) if response.data else []
    legacy = []
    for gpu in gpus:
        total_bytes = int((gpu.get("total_gb", 0) or 0) * 1024**3)
        used_bytes = int((gpu.get("allocated_gb", 0) or 0) * 1024**3)
        legacy.append({
            "index": gpu.get("id", 0),
            "name": gpu.get("name", "Unknown GPU"),
            "total_memory": total_bytes,
            "used_memory": used_bytes,
            "free_memory": max(total_bytes - used_bytes, 0),
            "utilization": gpu.get("utilization", 0),
        })
    return legacy


@router.get("/models")
async def list_models():
    """兼容旧前端的模型列表接口"""
    try:
        output_dir = PROJECT_ROOT / "output"
        model_paths: List[str] = []
        if output_dir.exists():
            patterns = ["*.safetensors", "*.ckpt", "*.pt", "*.pth"]
            for pattern in patterns:
                for file in output_dir.rglob(pattern):
                    if file.is_file():
                        model_paths.append(str(file))
        return model_paths
    except Exception as e:
        log.error(f"列出模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pick_folder", response_model=SystemResponse)
async def pick_folder():
    """打开文件夹选择对话框 (Windows)"""
    try:
        if sys.platform == "win32":
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()  # 隐藏主窗口
            root.attributes('-topmost', True)  # 置顶
            
            folder = filedialog.askdirectory()
            root.destroy()
            
            if folder:
                return SystemResponse(
                    status="success",
                    data={"path": folder}
                )
            else:
                return SystemResponse(status="cancelled", message="用户取消选择")
        else:
            # Linux/Mac 暂时返回空
            return SystemResponse(status="error", message="当前平台不支持文件选择对话框")
            
    except Exception as e:
        log.error(f"打开文件夹选择对话框失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pick_file", response_model=SystemResponse)
async def pick_file(file_type: str = "model"):
    """打开文件选择对话框"""
    try:
        if sys.platform == "win32":
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            if file_type == "model":
                filetypes = [
                    ("模型文件", "*.safetensors;*.ckpt;*.pt"),
                    ("所有文件", "*.*")
                ]
            elif file_type == "yaml":
                filetypes = [
                    ("YAML 文件", "*.yaml;*.yml"),
                    ("所有文件", "*.*")
                ]
            else:
                filetypes = [("所有文件", "*.*")]
            
            file = filedialog.askopenfilename(filetypes=filetypes)
            root.destroy()
            
            if file:
                return SystemResponse(
                    status="success",
                    data={"path": file}
                )
            else:
                return SystemResponse(status="cancelled", message="用户取消选择")
        else:
            return SystemResponse(status="error", message="当前平台不支持文件选择对话框")
            
    except Exception as e:
        log.error(f"打开文件选择对话框失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list_directory", response_model=SystemResponse)
async def list_directory(path: str = ""):
    """列出目录内容"""
    try:
        if not path:
            path = str(PROJECT_ROOT)
        
        target_path = Path(path)
        if not target_path.exists():
            raise HTTPException(status_code=404, detail=f"路径不存在: {path}")
        
        if not target_path.is_dir():
            raise HTTPException(status_code=400, detail=f"不是目录: {path}")
        
        items = []
        for item in sorted(target_path.iterdir()):
            stat = item.stat()
            items.append({
                "name": item.name,
                "path": str(item),
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else 0,
                "modified": stat.st_mtime
            })
        
        return SystemResponse(
            status="success",
            data={
                "path": str(target_path),
                "items": items
            }
        )
        
    except Exception as e:
        log.error(f"列出目录失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
