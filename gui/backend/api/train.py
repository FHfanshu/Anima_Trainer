"""
训练控制 API
提供训练启动、停止、状态监控等功能
"""

import asyncio
import subprocess
import sys
import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from gui.backend.log import log
from gui.backend.services.trainer import trainer_service

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class TrainStartRequest(BaseModel):
    config: Dict[str, Any]


class TrainResponse(BaseModel):
    status: str
    message: str = ""
    data: Optional[Dict[str, Any]] = None


class MetricsData(BaseModel):
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    lr: float = 0.0
    gpu_memory: float = 0.0
    timestamp: str = ""


# 存储训练指标 (用于图表)
train_metrics: List[MetricsData] = []


@router.post("/start", response_model=TrainResponse)
async def start_training(request: TrainStartRequest, background_tasks: BackgroundTasks):
    """启动训练任务"""
    try:
        config = request.config
        
        # 验证必要参数
        if not config.get("data_root"):
            raise HTTPException(status_code=400, detail="数据集路径不能为空")
        
        data_root = Path(config["data_root"])
        if not data_root.exists():
            raise HTTPException(status_code=400, detail=f"数据集路径不存在: {data_root}")
        
        # 保存配置为 YAML
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        config_dir = PROJECT_ROOT / "config" / "autosave"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / f"train_{timestamp}.yaml"
        
        import yaml
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        
        log.info(f"配置已保存: {config_file}")
        
        # 构建训练命令
        cmd = [
            sys.executable, "-m", "accelerate.commands.launch",
            "--num_cpu_threads_per_process", str(config.get("dataloader_num_workers", 4)),
            "--quiet",
            str(PROJECT_ROOT / "train.py"),
            "--config_file", str(config_file),
        ]
        
        # 设置环境变量
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"
        
        # 启动训练
        task_id = trainer_service.start_training(cmd, env)
        
        if not task_id:
            raise HTTPException(status_code=500, detail="无法创建训练任务")
        
        # 清空之前的指标
        train_metrics.clear()
        
        log.info(f"训练任务已启动，ID: {task_id}")
        
        return TrainResponse(
            status="success",
            message="训练已启动",
            data={"task_id": task_id, "config_file": str(config_file)}
        )
        
    except Exception as e:
        log.error(f"启动训练失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=TrainResponse)
async def stop_training():
    """停止训练任务"""
    try:
        trainer_service.stop_training()
        log.info("训练任务已停止")
        return TrainResponse(status="success", message="训练已停止")
    except Exception as e:
        log.error(f"停止训练失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=TrainResponse)
async def get_training_status():
    """获取训练状态"""
    try:
        status = trainer_service.get_status()
        return TrainResponse(
            status="success",
            data=status
        )
    except Exception as e:
        log.error(f"获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs", response_model=TrainResponse)
async def get_training_logs(lines: int = 100):
    """获取训练日志"""
    try:
        logs = trainer_service.get_logs(lines)
        return TrainResponse(
            status="success",
            data={"logs": logs}
        )
    except Exception as e:
        log.error(f"获取日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=TrainResponse)
async def get_training_metrics():
    """获取训练指标 (用于图表)"""
    try:
        # 解析最新日志更新指标
        _parse_latest_logs()
        
        return TrainResponse(
            status="success",
            data={
                "metrics": [m.dict() for m in train_metrics],
                "count": len(train_metrics)
            }
        )
    except Exception as e:
        log.error(f"获取指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/checkpoints", response_model=TrainResponse)
async def list_checkpoints():
    """列出所有 checkpoint"""
    try:
        output_dir = PROJECT_ROOT / "output"
        checkpoints = []
        
        if output_dir.exists():
            for ckpt_dir in sorted(output_dir.iterdir(), reverse=True):
                if ckpt_dir.is_dir():
                    # 检查是否有模型文件
                    model_files = list(ckpt_dir.glob("*.safetensors")) + list(ckpt_dir.glob("*.pt"))
                    if model_files:
                        stat = ckpt_dir.stat()
                        checkpoints.append({
                            "name": ckpt_dir.name,
                            "path": str(ckpt_dir),
                            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            "model_count": len(model_files)
                        })
        
        return TrainResponse(
            status="success",
            data={"checkpoints": checkpoints}
        )
    except Exception as e:
        log.error(f"列出 checkpoint 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _parse_latest_logs():
    """解析日志提取指标"""
    logs = trainer_service.get_logs(50)
    
    # 正则表达式匹配训练日志
    # 示例: "epoch: 12, step: 340, loss: 0.1423, lr: 1.2e-4"
    patterns = {
        'epoch': r'epoch[:=]\s*(\d+)',
        'step': r'step[:=]\s*(\d+)',
        'loss': r'loss[:=]\s*([\d.]+)',
        'lr': r'lr[:=]\s*([\d.e-]+)',
        'gpu': r'GPU[^\d]*(\d+\.?\d*)\s*MB',
    }
    
    for line in logs:
        # 尝试提取指标
        metrics = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    if key == 'gpu':
                        metrics['gpu_memory'] = float(match.group(1)) / 1024  # 转换为 GB
                    else:
                        metrics[key] = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                except:
                    pass
        
        # 如果有有效指标，添加到列表
        if 'loss' in metrics and 'step' in metrics:
            # 避免重复添加同一步
            if not train_metrics or train_metrics[-1].step != metrics.get('step', 0):
                train_metrics.append(MetricsData(
                    epoch=metrics.get('epoch', 0),
                    step=metrics.get('step', 0),
                    loss=metrics.get('loss', 0),
                    lr=metrics.get('lr', 0),
                    gpu_memory=metrics.get('gpu_memory', 0),
                    timestamp=datetime.now().isoformat()
                ))
                
                # 限制历史记录数量 (保留最近 2000 点)
                if len(train_metrics) > 2000:
                    train_metrics.pop(0)
