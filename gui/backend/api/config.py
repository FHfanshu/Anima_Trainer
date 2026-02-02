"""
配置管理 API
提供训练配置的 CRUD 操作
"""

import os
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gui.backend.log import log

router = APIRouter()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# 默认配置 (来自 train.py 的 TrainingConfig)
DEFAULT_CONFIG = {
    # 模型参数
    "pretrained_model_name_or_path": "circlestone-labs/Anima",
    "comfyui_path": None,
    "revision": None,
    "variant": None,
    
    # LoRA 参数
    "lora_type": "lora",
    "lora_rank": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lora_target_modules": [
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.net.0.proj", "ff.net.2",
        "proj_in", "proj_out",
    ],
    
    # LyCORIS LoKr 参数
    "lokr_factor": 8,
    "lokr_use_effective_conv2d": True,
    
    # 数据集参数
    "data_root": "./data",
    "resolution": 1024,
    "center_crop": True,
    "random_flip": True,
    "tag_dropout": 0.1,
    
    # 训练参数
    "output_dir": "./output",
    "seed": 42,
    "num_train_epochs": 100,
    "max_train_steps": None,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    
    # 优化器参数
    "optimizer": "adamw8bit",
    "learning_rate": 1e-4,
    "scale_lr": True,
    "lr_scheduler": "cosine_with_restarts",
    "lr_warmup_steps": 500,
    "lr_num_cycles": 1,
    "lr_power": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    
    # 精度配置
    "mixed_precision": "bf16",
    "enable_flash_attention": True,
    
    # Checkpoint 配置
    "checkpointing_steps": 500,
    "checkpoints_total_limit": 5,
    "resume_from_checkpoint": None,
    "save_state": True,
    
    # 验证配置
    "validation_prompt": None,
    "validation_epochs": 5,
    "num_validation_images": 4,
    
    # WandB 配置
    "report_to": "wandb",
    "wandb_project": "anima-lora-training",
    "wandb_entity": None,
    "tracker_run_name": None,
    
    # 其他
    "dataloader_num_workers": 4,
    "min_snr_gamma": 5.0,
}

# 预设配置
PRESETS = {
    "character": {
        "name": "角色训练",
        "description": "适合训练特定动漫角色",
        "config": {
            "lora_rank": 32,
            "lora_alpha": 32,
            "num_train_epochs": 100,
            "learning_rate": 1e-4,
            "tag_dropout": 0.1,
            "resolution": 1024,
        }
    },
    "style": {
        "name": "画风训练",
        "description": "适合训练特定艺术风格",
        "config": {
            "lora_rank": 64,
            "lora_alpha": 64,
            "num_train_epochs": 150,
            "learning_rate": 5e-5,
            "tag_dropout": 0.2,
            "resolution": 1024,
        }
    },
    "concept": {
        "name": "概念训练",
        "description": "适合训练特定概念或物体",
        "config": {
            "lora_rank": 16,
            "lora_alpha": 16,
            "num_train_epochs": 80,
            "learning_rate": 2e-4,
            "tag_dropout": 0.05,
            "resolution": 1024,
        }
    },
    "quick": {
        "name": "快速测试",
        "description": "快速验证配置是否正确",
        "config": {
            "lora_rank": 16,
            "lora_alpha": 16,
            "num_train_epochs": 5,
            "max_train_steps": 100,
            "learning_rate": 1e-4,
            "checkpointing_steps": 50,
            "validation_epochs": 1,
        }
    }
}


class ConfigResponse(BaseModel):
    status: str
    message: str = ""
    data: Optional[Dict[str, Any]] = None


@router.get("/default", response_model=ConfigResponse)
async def get_default_config():
    """获取默认配置"""
    return ConfigResponse(
        status="success",
        data=DEFAULT_CONFIG.copy()
    )


@router.get("/presets", response_model=ConfigResponse)
async def get_presets():
    """获取可用预设列表"""
    return ConfigResponse(
        status="success",
        data=PRESETS
    )


@router.post("/load_preset/{preset_name}", response_model=ConfigResponse)
async def load_preset(preset_name: str):
    """加载预设配置"""
    if preset_name not in PRESETS:
        raise HTTPException(status_code=404, detail=f"预设 '{preset_name}' 不存在")
    
    preset = PRESETS[preset_name]
    config = DEFAULT_CONFIG.copy()
    config.update(preset["config"])
    
    return ConfigResponse(
        status="success",
        message=f"已加载预设: {preset['name']}",
        data=config
    )


@router.post("/save", response_model=ConfigResponse)
async def save_config(config: Dict[str, Any]):
    """保存配置到文件"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        config_dir = PROJECT_ROOT / "config" / "autosave"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / f"{timestamp}.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        
        log.info(f"配置已保存: {config_file}")
        
        return ConfigResponse(
            status="success",
            message=f"配置已保存: {config_file.name}",
            data={"path": str(config_file)}
        )
    except Exception as e:
        log.error(f"保存配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")


@router.get("/list", response_model=ConfigResponse)
async def list_saved_configs():
    """列出所有保存的配置文件"""
    try:
        config_dir = PROJECT_ROOT / "config" / "autosave"
        if not config_dir.exists():
            return ConfigResponse(status="success", data=[])
        
        configs = []
        for config_file in sorted(config_dir.glob("*.yaml"), reverse=True):
            stat = config_file.stat()
            configs.append({
                "name": config_file.stem,
                "path": str(config_file),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return ConfigResponse(status="success", data=configs)
    except Exception as e:
        log.error(f"列出配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load/{config_name}", response_model=ConfigResponse)
async def load_config(config_name: str):
    """加载指定配置文件"""
    try:
        config_file = PROJECT_ROOT / "config" / "autosave" / f"{config_name}.yaml"
        
        if not config_file.exists():
            raise HTTPException(status_code=404, detail=f"配置文件 '{config_name}' 不存在")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return ConfigResponse(
            status="success",
            message=f"已加载配置: {config_name}",
            data=config
        )
    except Exception as e:
        log.error(f"加载配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{config_name}", response_model=ConfigResponse)
async def delete_config(config_name: str):
    """删除配置文件"""
    try:
        config_file = PROJECT_ROOT / "config" / "autosave" / f"{config_name}.yaml"
        
        if not config_file.exists():
            raise HTTPException(status_code=404, detail=f"配置文件 '{config_name}' 不存在")
        
        config_file.unlink()
        log.info(f"配置已删除: {config_file}")
        
        return ConfigResponse(
            status="success",
            message=f"配置已删除: {config_name}"
        )
    except Exception as e:
        log.error(f"删除配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
