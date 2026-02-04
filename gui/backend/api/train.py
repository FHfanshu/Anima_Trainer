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
from gui.backend.api.config_normalize import normalize_training_config

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
    """启动训练任务 (anima_train.py 作为唯一入口)"""
    try:
        config = normalize_training_config(request.config)

        def as_dict(value: Any) -> Dict[str, Any]:
            return value if isinstance(value, dict) else {}

        network_cfg = as_dict(config.get("network"))
        caption_cfg = as_dict(config.get("caption"))

        def resolve_required_dir(value: Any, error_label: str) -> Path:
            if not value:
                raise HTTPException(status_code=400, detail=f"{error_label}不能为空")
            path = Path(str(value)).expanduser()
            if not path.exists():
                raise HTTPException(status_code=400, detail=f"{error_label}不存在: {path}")
            return path.resolve()

        def resolve_required_file(value: Any, error_label: str) -> Path:
            if not value:
                raise HTTPException(status_code=400, detail=f"{error_label}不能为空")
            path = Path(str(value)).expanduser()
            if not path.exists():
                raise HTTPException(status_code=400, detail=f"{error_label}不存在: {path}")
            if path.is_dir():
                raise HTTPException(status_code=400, detail=f"{error_label}应为文件: {path}")
            return path.resolve()

        def resolve_optional_file(value: Any) -> Optional[Path]:
            if not value:
                return None
            path = Path(str(value)).expanduser()
            if not path.exists() or path.is_dir():
                return None
            return path.resolve()

        def resolve_optional_dir(value: Any) -> Optional[Path]:
            if not value:
                return None
            path = Path(str(value)).expanduser()
            if not path.exists() or not path.is_dir():
                return None
            return path.resolve()

        def pick_default_path(candidates: List[Path]) -> Optional[Path]:
            for cand in candidates:
                if cand.exists():
                    return cand
            return None

        data_root = resolve_required_dir(config.get("data_root"), "数据集路径")

        default_transformer = PROJECT_ROOT / "models" / "transformers" / "anima-preview.safetensors"
        default_vae = PROJECT_ROOT / "models" / "vae" / "qwen_image_vae.safetensors"
        default_qwen = PROJECT_ROOT / "models" / "text_encoders"

        transformer_value = config.get("transformer_path") or config.get("transformer") or config.get("pretrained_model_name_or_path")
        transformer = None
        if transformer_value:
            transformer = resolve_required_file(transformer_value, "Transformer 权重")
        else:
            transformer = pick_default_path([default_transformer])
            if not transformer:
                raise HTTPException(status_code=400, detail="Transformer 权重不能为空")

        vae_value = config.get("vae_path") or config.get("vae")
        vae = None
        if vae_value:
            vae = resolve_required_file(vae_value, "VAE 权重")
        else:
            vae = pick_default_path([default_vae])
            if not vae:
                raise HTTPException(status_code=400, detail="VAE 权重不能为空")

        qwen_value = config.get("qwen_path") or config.get("qwen") or config.get("text_encoder_path")
        qwen_dir = None
        if qwen_value:
            qwen_dir = resolve_required_dir(qwen_value, "Qwen 模型目录")
        else:
            qwen_dir = pick_default_path([default_qwen])
            if not qwen_dir:
                raise HTTPException(status_code=400, detail="Qwen 模型目录不能为空")

        t5_tokenizer = resolve_optional_dir(config.get("t5_tokenizer_path") or config.get("t5_tokenizer"))

        output_dir = Path(str(config.get("output_dir") or "./output")).expanduser().resolve()
        output_name = config.get("output_name") or "anima_lora"

        cmd: List[str] = [sys.executable, str(PROJECT_ROOT / "anima_train.py")]

        def add_arg(flag: str, value: Any):
            if value is None or value == "":
                return
            cmd.extend([flag, str(value)])

        def add_flag(flag: str, value: Any):
            if bool(value):
                cmd.append(flag)

        add_arg("--data-dir", data_root)
        add_arg("--transformer", transformer)
        add_arg("--vae", vae)
        add_arg("--qwen", qwen_dir)
        if t5_tokenizer:
            add_arg("--t5-tokenizer", t5_tokenizer)

        add_arg("--output-dir", output_dir)
        add_arg("--output-name", output_name)
        add_arg("--lora-name", config.get("lora_name"))

        add_arg("--epochs", config.get("num_train_epochs") or config.get("epochs"))
        add_arg("--batch-size", config.get("train_batch_size") or config.get("batch_size"))
        add_arg("--grad-accum", config.get("gradient_accumulation_steps") or config.get("grad_accum"))
        add_arg("--lr", config.get("learning_rate") or config.get("lr"))
        add_arg("--resolution", config.get("resolution"))
        add_arg("--max-steps", config.get("max_train_steps") or config.get("max_steps"))
        add_arg("--num-workers", config.get("dataloader_num_workers") or config.get("num_workers"))

        mp = config.get("mixed_precision")
        if mp:
            mp = str(mp).lower()
            if mp == "no":
                mp = "fp32"
            if mp not in ("fp32", "fp16", "bf16"):
                mp = "bf16"
            add_arg("--mixed-precision", mp)

        add_flag("--grad-checkpoint", config.get("gradient_checkpointing") or config.get("grad_checkpoint"))
        add_flag("--xformers", config.get("enable_xformers") or config.get("xformers"))

        add_arg("--repeats", config.get("repeats"))
        add_flag("--shuffle-caption", config.get("shuffle_caption") or caption_cfg.get("shuffle_caption"))
        add_arg("--keep-tokens", config.get("keep_tokens") or caption_cfg.get("keep_tokens"))
        add_flag("--flip-augment", config.get("random_flip") or config.get("flip_augment"))
        add_flag("--cache-latents", config.get("cache_latents"))

        lora_type = config.get("lora_type") or network_cfg.get("network_type")
        add_arg("--lora-type", lora_type)
        add_arg("--lora-rank", config.get("lora_rank") or network_cfg.get("network_dim"))
        add_arg("--lora-alpha", config.get("lora_alpha") or network_cfg.get("network_alpha"))
        add_arg("--lokr-factor", config.get("lokr_factor") or network_cfg.get("lokr_factor"))

        add_arg("--save-every", config.get("save_every") or config.get("save_every_n_epochs"))
        save_state_value = config.get("save_last_n_epochs_state") or config.get("save_state")
        if isinstance(save_state_value, bool) and save_state_value:
            add_arg("--save-state", 1)
        elif save_state_value:
            add_arg("--save-state", save_state_value)

        add_arg("--resume", config.get("resume_from_checkpoint") or config.get("resume"))
        add_arg("--seed", config.get("seed"))

        add_arg("--sample-every", config.get("sample_every"))
        add_arg("--sample-prompt", config.get("sample_prompt"))

        add_arg("--loss-curve-steps", config.get("loss_curve_steps"))
        add_flag("--no-progress", config.get("no_progress"))
        add_flag("--auto-install", config.get("auto_install"))

        # 保存配置为 YAML (原始 GUI 配置)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        config_dir = PROJECT_ROOT / "config" / "autosave"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / f"train_{timestamp}.yaml"

        import yaml
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        log.info(f"配置已保存: {config_file}")
        log.info(f"训练命令: {' '.join(cmd)}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

        task_id = trainer_service.start_training(cmd, env)

        if not task_id:
            raise HTTPException(status_code=500, detail="无法创建训练任务")

        train_metrics.clear()

        log.info(f"训练任务已启动，ID: {task_id}")

        return TrainResponse(
            status="success",
            message="训练已启动",
            data={"task_id": task_id, "config_file": str(config_file)}
        )

    except HTTPException:
        raise
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
