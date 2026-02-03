"""
Anima Character LoRA 训练脚本
==============================
这是一个完整的、可用于生产环境的 LoRA/LyCORIS Lokr 训练脚本，
专门为训练 circlestone-labs/Anima 的 Character LoRA 设计。

特性：
- 支持 8-bit AdamW (bitsandbytes) 优化器
- 支持 gradient checkpointing 和 Flash Attention 2
- 支持 tag-based 数据集
- 完整的 WandB 日志和 checkpoint 管理
- 支持从 checkpoint 恢复训练
- 兼容 RTX 3090 单卡训练

作者: PyTorch 工程师
日期: 2026-02-02
"""

import os
import sys
import argparse
import logging
import math
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, fields
import gc
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Diffusers 相关
from diffusers import (
    AutoencoderKLCosmos,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
    DiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.training_utils import EMAModel, cast_training_params

# PEFT 相关 (LoRA)
from peft import LoraConfig, get_peft_model, PeftModel
from peft.utils import get_peft_model_state_dict

# Accelerate 相关
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    ProjectConfiguration,
    set_seed,
    DistributedDataParallelKwargs,
)

# Transformers 相关
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PretrainedConfig,
)

# 其他工具
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from omegaconf import OmegaConf

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - fallback for Python 3.10
    tomllib = None

try:
    import tomli  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tomli = None

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

# 导入自定义模块
from utils.dataset import TagBasedDataset, CachedTagBasedDataset, BucketBatchSampler, collate_fn
from utils.model_utils import load_anima_pipeline, setup_lora_adapters
from utils.optimizer_utils import create_optimizer
from utils.checkpoint import CheckpointManager
from utils.comfyui_loader import load_anima_from_comfyui, load_anima_with_fallback

# 检查 diffusers 最低版本
check_min_version("0.32.0")

# 获取 logger
logger = get_logger(__name__, log_level="INFO")


def configure_torch_backends(enable_flash_attention: bool) -> None:
    """Configure PyTorch backends for speed.

    Notes:
    - Cosmos uses PyTorch SDPA (scaled_dot_product_attention). On CUDA, PyTorch can select
      Flash-SDP / MemEff-SDP kernels when enabled and supported.
    - This is independent from the external `flash-attn` package.
    """

    if not torch.cuda.is_available():
        return

    # TF32 can speed up some fp32 matmuls/convs on Ampere+.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Prefer faster SDPA kernels when possible.
    try:
        if enable_flash_attention:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        else:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


@dataclass
class TrainingConfig:
    """
    训练配置类
    使用 dataclass 来管理所有训练参数，便于类型检查和默认值管理
    """
    # 模型参数
    pretrained_model_name_or_path: str = field(
        default="circlestone-labs/Anima",
        metadata={"help": "预训练模型的 Hugging Face 路径或本地路径"}
    )
    comfyui_path: Optional[str] = field(
        default=None,
        metadata={"help": "ComfyUI 安装路径，从 ComfyUI/models 加载模型 (优先级高于 HF 路径)"}
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "模型版本修订号"}
    )
    variant: Optional[str] = field(
        default=None,
        metadata={"help": "模型变体，如 'fp16'"}
    )
    
    # LoRA 参数
    lora_type: str = field(
        default="lora",
        metadata={"help": "LoRA 类型: 'lora' (标准PEFT) 或 'lokr' (LyCORIS)"}
    )
    lora_rank: int = field(
        default=32,
        metadata={"help": "LoRA rank，控制可训练参数数量"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha，控制 LoRA 强度"}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout 概率"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "to_q", "to_k", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
            "proj_in", "proj_out",
        ],
        metadata={"help": "要应用 LoRA 的目标模块列表"}
    )
    
    # LyCORIS LoKr 专用参数
    lokr_factor: int = field(
        default=8,
        metadata={"help": "LoKr 分解因子"}
    )
    lokr_use_effective_conv2d: bool = field(
        default=True,
        metadata={"help": "LoKr 是否使用有效的 conv2d"}
    )
    
    # 数据集参数
    data_root: str = field(
        default="./data",
        metadata={"help": "数据集根目录，包含图片和对应的 .txt 标签文件"}
    )
    resolution: int = field(
        default=1024,
        metadata={"help": "训练图像分辨率"}
    )
    center_crop: bool = field(
        default=True,
        metadata={"help": "是否中心裁剪图像"}
    )
    random_flip: bool = field(
        default=True,
        metadata={"help": "是否随机水平翻转图像用于数据增强"}
    )

    # Latent 缓存（默认开启：只训练 DiT/UNet，节省 VAE 编码时间）
    cache_latents: bool = field(
        default=True,
        metadata={"help": "是否预先缓存 VAE latent（更快，但会占用内存）"}
    )

    # ARB (aspect ratio bucketing)
    enable_arb: bool = field(
        default=False,
        metadata={"help": "是否启用 ARB（按长宽比分桶训练）"}
    )
    min_bucket_reso: int = field(
        default=512,
        metadata={"help": "ARB 最小桶分辨率（像素，宽/高均在该范围内）"}
    )
    max_bucket_reso: int = field(
        default=1024,
        metadata={"help": "ARB 最大桶分辨率（像素，宽/高均在该范围内）"}
    )
    bucket_reso_steps: int = field(
        default=64,
        metadata={"help": "ARB 桶分辨率步进（建议 64；会对齐到 8）"}
    )
    bucket_no_upscale: bool = field(
        default=False,
        metadata={"help": "ARB 是否避免把小图 upscale 到更大桶（可能导致桶选择受限）"}
    )
    
    # 训练参数
    output_dir: str = field(
        default="./output",
        metadata={"help": "输出目录，用于保存模型和日志"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "随机种子，用于结果可复现"}
    )
    num_train_epochs: int = field(
        default=100,
        metadata={"help": "训练 epoch 数量"}
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "最大训练步数，如果设置则覆盖 num_train_epochs"}
    )
    train_batch_size: int = field(
        default=1,
        metadata={"help": "每个设备的训练批次大小"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "梯度累积步数，用于模拟更大的 batch size"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "是否启用梯度检查点，显著减少显存占用"}
    )
    
    # 优化器参数
    optimizer: str = field(
        default="adamw8bit",
        metadata={"help": "优化器类型: 'adamw8bit' (8-bit AdamW) 或 'adamw'"}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "学习率"}
    )
    scale_lr: bool = field(
        default=True,
        metadata={"help": "是否根据 batch size 和 GPU 数量缩放学习率"}
    )
    lr_scheduler: str = field(
        default="cosine_with_restarts",
        metadata={"help": "学习率调度器类型"}
    )
    lr_warmup_steps: int = field(
        default=500,
        metadata={"help": "学习率预热步数"}
    )
    lr_num_cycles: int = field(
        default=1,
        metadata={"help": "余弦调度器的周期数"}
    )
    lr_power: float = field(
        default=1.0,
        metadata={"help": "多项式调度器的幂指数"}
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Adam beta1 参数"}
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Adam beta2 参数"}
    )
    adam_weight_decay: float = field(
        default=1e-2,
        metadata={"help": "Adam weight decay 参数"}
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Adam epsilon 参数"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "梯度裁剪的最大范数"}
    )
    
    # 精度参数
    mixed_precision: str = field(
        default="bf16",
        metadata={"help": "混合精度训练: 'no', 'fp16', 'bf16'"}
    )
    
    # Flash Attention 参数
    enable_flash_attention: bool = field(
        default=True,
        metadata={"help": "是否启用 Flash Attention 2"}
    )
    
    # Checkpoint 参数
    checkpointing_steps: int = field(
        default=500,
        metadata={"help": "每多少 step 保存一次 checkpoint（0 表示禁用）"}
    )
    checkpointing_epochs: int = field(
        default=0,
        metadata={"help": "每多少个 epoch 保存一次 checkpoint（0 表示禁用）"}
    )
    checkpoints_total_limit: Optional[int] = field(
        default=None,
        metadata={"help": "保留的 checkpoint 最大数量"}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "从指定 checkpoint 路径恢复训练"}
    )
    
    # 验证参数
    validation_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "验证时使用的提示词"}
    )
    validation_epochs: int = field(
        default=5,
        metadata={"help": "每多少个 epoch 进行一次验证"}
    )
    num_validation_images: int = field(
        default=4,
        metadata={"help": "每次验证生成的图像数量"}
    )
    
    # WandB 参数
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "报告工具: 'tensorboard', 'wandb', 'all', 'none'"}
    )
    wandb_project: str = field(
        default="anima-lora-training",
        metadata={"help": "WandB 项目名称"}
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "WandB 实体（团队或个人）名称"}
    )
    tracker_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "WandB run 名称"}
    )
    
    # 其他参数
    num_workers: int = field(
        default=4,
        metadata={"help": "数据加载的 worker 数量"}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={"help": "DataLoader 的 worker 数量"}
    )
    save_state: bool = field(
        default=True,
        metadata={"help": "是否保存完整训练状态（包括优化器状态）"}
    )
    tag_dropout: float = field(
        default=0.1,
        metadata={"help": "tag dropout 概率，增强泛化能力"}
    )
    min_snr_gamma: float = field(
        default=5.0,
        metadata={"help": "Min-SNR 加权策略的 gamma 参数"}
    )
    

def parse_args() -> TrainingConfig:
    """
    解析命令行参数并返回 TrainingConfig 对象
    
    为什么使用 argparse：
    1. 提供命令行接口，便于脚本化和自动化
    2. 自动生成帮助文档
    3. 类型检查和默认值管理
    4. 支持从配置文件加载参数
    """
    parser = argparse.ArgumentParser(
        description="Anima Character LoRA Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to a TOML/YAML config file"
    )
    
    # 模型参数
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="circlestone-labs/Anima",
        help="预训练模型的 Hugging Face 路径或本地路径"
    )
    parser.add_argument(
        "--comfyui_path",
        type=str,
        default=None,
        help="ComfyUI 安装路径，从 ComfyUI/models 加载模型 (优先级高于 HF 路径)"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="模型版本修订号"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="模型变体，如 'fp16'"
    )
    
    # LoRA 参数
    parser.add_argument(
        "--lora_type",
        type=str,
        default="lora",
        choices=["lora", "lokr"],
        help="LoRA 类型: 'lora' (标准PEFT) 或 'lokr' (LyCORIS)"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank，控制可训练参数数量"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha，控制 LoRA 强度"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout 概率"
    )
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        help="要应用 LoRA 的目标模块列表"
    )
    
    # LyCORIS LoKr 专用参数
    parser.add_argument(
        "--lokr_factor",
        type=int,
        default=8,
        help="LoKr 分解因子"
    )
    parser.add_argument(
        "--lokr_use_effective_conv2d",
        action="store_true",
        default=True,
        help="LoKr 是否使用有效的 conv2d"
    )
    
    # 数据集参数
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="数据集根目录，包含图片和对应的 .txt 标签文件"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="训练图像分辨率"
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        default=True,
        help="是否中心裁剪图像"
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        default=True,
        help="是否随机水平翻转图像用于数据增强"
    )

    # ARB (aspect ratio bucketing)
    parser.add_argument(
        "--enable_arb",
        action="store_true",
        default=False,
        help="启用 ARB（按长宽比分桶训练）"
    )
    parser.add_argument(
        "--min_bucket_reso",
        type=int,
        default=512,
        help="ARB 最小桶分辨率（像素）"
    )
    parser.add_argument(
        "--max_bucket_reso",
        type=int,
        default=1024,
        help="ARB 最大桶分辨率（像素）"
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="ARB 桶分辨率步进（建议 64；会对齐到 8）"
    )
    parser.add_argument(
        "--bucket_no_upscale",
        action="store_true",
        default=False,
        help="ARB 避免 upscale（更接近 kohya 的 no_upscale 行为）"
    )
    parser.add_argument(
        "--no_center_crop",
        action="store_false",
        dest="center_crop",
        help="禁用中心裁剪"
    )
    parser.add_argument(
        "--no_random_flip",
        action="store_false",
        dest="random_flip",
        help="禁用随机翻转"
    )

    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=True,
        help="缓存 VAE latent（默认开启）"
    )
    parser.add_argument(
        "--no_cache_latents",
        action="store_false",
        dest="cache_latents",
        help="禁用 latent 缓存（会在每步运行 VAE encode）"
    )
    
    # 训练参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="输出目录，用于保存模型和日志"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于结果可复现"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="训练 epoch 数量"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="最大训练步数，如果设置则覆盖 num_train_epochs"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="每个设备的训练批次大小"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="梯度累积步数，用于模拟更大的 batch size"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="是否启用梯度检查点，显著减少显存占用"
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
        help="禁用梯度检查点"
    )
    
    # 优化器参数
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw8bit",
        choices=["adamw8bit", "adamw"],
        help="优化器类型: 'adamw8bit' (8-bit AdamW) 或 'adamw'"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="学习率"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="是否根据 batch size 和 GPU 数量缩放学习率"
    )
    parser.add_argument(
        "--no_scale_lr",
        action="store_false",
        dest="scale_lr",
        help="禁用学习率缩放"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_with_restarts",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="学习率调度器类型"
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="学习率预热步数"
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="余弦调度器的周期数"
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="多项式调度器的幂指数"
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1 参数"
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 参数"
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Adam weight decay 参数"
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Adam epsilon 参数"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪的最大范数"
    )
    
    # 精度参数
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="混合精度训练"
    )
    
    # Flash Attention 参数
    parser.add_argument(
        "--enable_flash_attention",
        action="store_true",
        default=True,
        help="是否启用 Flash Attention 2"
    )
    parser.add_argument(
        "--disable_flash_attention",
        action="store_false",
        dest="enable_flash_attention",
        help="禁用 Flash Attention 2"
    )
    
    # Checkpoint 参数
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="每多少 step 保存一次 checkpoint（0 表示禁用）"
    )
    parser.add_argument(
        "--checkpointing_epochs",
        type=int,
        default=0,
        help="每多少个 epoch 保存一次 checkpoint（0 表示禁用）"
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="保留的 checkpoint 最大数量"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从指定 checkpoint 路径恢复训练"
    )
    
    # 验证参数
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="验证时使用的提示词"
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="每多少个 epoch 进行一次验证"
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="每次验证生成的图像数量"
    )
    
    # WandB 参数
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "all", "none"],
        help="报告工具"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="anima-lora-training",
        help="WandB 项目名称"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB 实体（团队或个人）名称"
    )
    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default=None,
        help="WandB run 名称"
    )
    
    # 其他参数
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="DataLoader 的 worker 数量"
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        default=True,
        help="是否保存完整训练状态（包括优化器状态）"
    )
    parser.add_argument(
        "--no_save_state",
        action="store_false",
        dest="save_state",
        help="不保存完整训练状态"
    )
    parser.add_argument(
        "--tag_dropout",
        type=float,
        default=0.1,
        help="tag dropout 概率，增强泛化能力"
    )
    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=5.0,
        help="Min-SNR 加权策略的 gamma 参数"
    )
    
    args, _ = parser.parse_known_args()

    config_overrides: Dict[str, Any] = {}
    if args.config_file:
        config_overrides = _load_config_file(args.config_file)
        config_overrides = _filter_config_overrides(config_overrides)
        parser.set_defaults(**config_overrides)

    args = parser.parse_args()

    # 转换为 TrainingConfig
    args_dict = vars(args)
    args_dict.pop("config_file", None)
    config = TrainingConfig(**args_dict)
    
    return config


def _training_config_keys() -> set[str]:
    return {field_obj.name for field_obj in fields(TrainingConfig)}


def _filter_config_overrides(config_data: Dict[str, Any]) -> Dict[str, Any]:
    valid_keys = _training_config_keys()
    unknown_keys = sorted(set(config_data.keys()) - valid_keys)
    if unknown_keys:
        print(f"[WARN] Ignoring unknown config keys: {', '.join(unknown_keys)}")
    return {key: value for key, value in config_data.items() if key in valid_keys}


def _load_config_file(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("YAML support not available. Install PyYAML or use TOML.")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif suffix == ".toml":
        toml_reader = tomllib or tomli
        if toml_reader is None:
            raise RuntimeError("TOML support not available. Use Python 3.11+ or install tomli.")
        with path.open("rb") as f:
            data = toml_reader.load(f)
    else:
        raise ValueError("Unsupported config file format. Use .toml or .yaml/.yml.")

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping/object.")

    return data


def main():
    """
    主训练函数
    这是整个训练流程的入口点，包含完整的训练逻辑
    """
    # =========================================================================
    # 第一步：解析参数
    # =========================================================================
    # 为什么先解析参数：确保所有配置在初始化前就位
    config = parse_args()

    configure_torch_backends(enable_flash_attention=config.enable_flash_attention)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # =========================================================================
    # 第二步：初始化 Accelerator
    # =========================================================================
    # Accelerator 是 Hugging Face 提供的分布式训练抽象层
    # 它自动处理：
    # - 混合精度训练 (fp16/bf16)
    # - 多 GPU 训练 (DDP)
    # - 梯度累积
    # - 设备放置 (CPU/GPU)
    # 
    # 使用 Accelerator 的好处：
    # 1. 代码可以在单卡和多卡之间无缝切换
    # 2. 自动处理梯度同步
    # 3. 提供统一的 API 接口
    
    logging_dir = os.path.join(config.output_dir, "logs")
    
    # 项目配置，用于 checkpoint 管理
    project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=logging_dir,
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=project_config,
    )
    
    # =========================================================================
    # 第三步：设置随机种子
    # =========================================================================
    # 随机种子确保实验可复现
    # 在多 GPU 训练中，每个进程需要不同的种子以避免完全相同的随机性
    if config.seed is not None:
        set_seed(config.seed)
        # 为每个进程设置不同的种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed + accelerator.process_index)
    
    # =========================================================================
    # 第四步：初始化日志
    # =========================================================================
    # 配置日志级别和格式
    # 只有主进程才打印详细日志，避免多进程重复输出
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Accelerator state: {accelerator.state}")
    logger.info(f"Process index: {accelerator.process_index}")
    logger.info(f"Local process index: {accelerator.local_process_index}")
    logger.info(f"World size: {accelerator.num_processes}")
    
    # 如果使用 wandb，只在主进程初始化
    if accelerator.is_main_process and config.report_to:
        # Filter config for tensorboard compatibility (only scalar types allowed)
        tracker_config = {
            k: v for k, v in vars(config).items()
            if isinstance(v, (int, float, str, bool)) and v is not None
        }
        project_name = config.wandb_project or "anima-lora-training"
        accelerator.init_trackers(
            project_name,
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    "entity": config.wandb_entity,
                    "name": config.tracker_run_name or f"anima-lora-r{config.lora_rank}-a{config.lora_alpha}",
                }
            }
        )
    
    # =========================================================================
    # 第五步：加载模型
    # =========================================================================
    # Anima 是基于 Cosmos 架构的，使用：
    # - CosmosTransformer3DModel: 核心扩散模型（替代 UNet）
    # - AutoencoderKLCosmos: VAE（基于 Qwen-Image VAE）
    # - 文本编码器：需要检查 Anima 的具体配置
    #
    # 模型加载顺序：
    # 1. VAE（用于编码/解码图像）
    # 2. Transformer（核心扩散模型）
    # 3. 文本编码器（编码文本提示）
    # 4. 噪声调度器（控制扩散过程）
    
    logger.info("Loading models...")
    
    # 确定数据类型
    # bf16 是 Ampere GPU（RTX 3090）的最佳选择，比 fp16 更稳定
    weight_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float32
    if config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    
    # 加载完整 Pipeline
    # 如果指定了 comfyui_path，优先从 ComfyUI 目录加载
    if config.comfyui_path:
        logger.info(f"Loading model from ComfyUI: {config.comfyui_path}")
        pipeline = load_anima_from_comfyui(
            comfyui_path=config.comfyui_path,
            torch_dtype=weight_dtype,
            device=accelerator.device,
            enable_flash_attention=config.enable_flash_attention,
        )
    else:
        # 从 HuggingFace 加载
        logger.info(f"Loading model from HuggingFace: {config.pretrained_model_name_or_path}")
        pipeline = load_anima_pipeline(
            config.pretrained_model_name_or_path,
            revision=config.revision,
            variant=config.variant,
            torch_dtype=weight_dtype,
            enable_flash_attention=config.enable_flash_attention,
        )
    
    # 提取各个组件
    transformer = pipeline.transformer
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    scheduler = pipeline.scheduler
    
    # =========================================================================
    # 第六步：设置 LoRA 适配器
    # =========================================================================
    # LoRA (Low-Rank Adaptation) 是一种参数高效微调方法
    # 它只训练少量低秩矩阵，而不是整个模型
    #
    # 优势：
    # 1. 大幅减少可训练参数（通常只有原模型的 1-5%）
    # 2. 训练速度快
    # 3. 存储成本低
    # 4. 可以与基础模型权重合并或分离
    #
    # 我们支持两种 LoRA 变体：
    # 1. 标准 PEFT LoRA：兼容性最好，生态最完善
    # 2. LyCORIS LoKr：更灵活，支持更多配置
    
    logger.info(f"Setting up {config.lora_type} adapters...")
    
    transformer = setup_lora_adapters(
        model=transformer,
        lora_type=config.lora_type,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        lokr_factor=config.lokr_factor if config.lora_type == "lokr" else None,
        lokr_use_effective_conv2d=config.lokr_use_effective_conv2d if config.lora_type == "lokr" else None,
    )
    
    # 打印可训练参数信息
    if accelerator.is_main_process:
        if hasattr(transformer, "print_trainable_parameters"):
            transformer.print_trainable_parameters()
        else:
            trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in transformer.parameters())
            logger.info(
                f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
            )
    
    # =========================================================================
    # 第七步：启用梯度检查点
    # =========================================================================
    # 梯度检查点（Gradient Checkpointing）是一种内存优化技术
    # 原理：在前向传播时不保存中间激活值，而是在反向传播时重新计算
    # 牺牲计算时间换取显存空间，通常可以减少 40-60% 的显存占用
    #
    # 注意：必须要在添加 LoRA 适配器后启用，否则可能不兼容
    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")
    
    # =========================================================================
    # 第八步：准备数据集和数据加载器
    # =========================================================================
    # Anima 使用 Danbooru 风格的 tag 格式
    # 数据集结构：
    # data_root/
    #   ├── image1.jpg
    #   ├── image1.txt
    #   ├── image2.png
    #   ├── image2.txt
    #   └── ...
    #
    # .txt 文件内容示例：
    # 1girl, oomuro sakurako, yuru yuri, brown hair, long hair, smile, ...
    
    logger.info("Loading dataset...")
    
    if config.cache_latents:
        train_dataset = CachedTagBasedDataset(
            data_root=config.data_root,
            tokenizer=tokenizer,
            vae=vae,
            device=accelerator.device,
            resolution=config.resolution,
            center_crop=config.center_crop,
            random_flip=config.random_flip,
            tag_dropout=config.tag_dropout,
            enable_arb=config.enable_arb,
            min_bucket_reso=config.min_bucket_reso,
            max_bucket_reso=config.max_bucket_reso,
            bucket_reso_steps=config.bucket_reso_steps,
            bucket_no_upscale=config.bucket_no_upscale,
        )
        logger.info("Using cached-latents dataset")
    else:
        train_dataset = TagBasedDataset(
            data_root=config.data_root,
            tokenizer=tokenizer,
            resolution=config.resolution,
            center_crop=config.center_crop,
            random_flip=config.random_flip,
            tag_dropout=config.tag_dropout,
            enable_arb=config.enable_arb,
            min_bucket_reso=config.min_bucket_reso,
            max_bucket_reso=config.max_bucket_reso,
            bucket_reso_steps=config.bucket_reso_steps,
            bucket_no_upscale=config.bucket_no_upscale,
        )

    if config.enable_arb and accelerator.num_processes == 1:
        batch_sampler = BucketBatchSampler(
            train_dataset,
            batch_size=config.train_batch_size,
            drop_last=True,
            shuffle=True,
            seed=config.seed,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=config.dataloader_num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=2 if config.dataloader_num_workers > 0 else None,
            persistent_workers=config.dataloader_num_workers > 0,
        )
        logger.info("ARB enabled: using bucketed batch sampler")
    else:
        if config.enable_arb and accelerator.num_processes != 1:
            logger.warning("ARB is enabled but multiple processes are active; falling back to non-ARB dataloader")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            prefetch_factor=2 if config.dataloader_num_workers > 0 else None,
            persistent_workers=config.dataloader_num_workers > 0,
        )
    
    logger.info(f"Dataset loaded: {len(train_dataset)} samples")
    
    # =========================================================================
    # 第九步：计算训练步数
    # =========================================================================
    # 需要计算总训练步数来设置学习率调度器
    # 公式：总步数 = 样本数 / (batch_size * gradient_accumulation_steps * num_processes)
    # 然后乘以 epoch 数
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    else:
        config.num_train_epochs = math.ceil(
            config.max_train_steps / num_update_steps_per_epoch
        )
    
    logger.info(f"Training steps: {config.max_train_steps}")
    logger.info(f"Training epochs: {config.num_train_epochs}")
    
    # =========================================================================
    # 第十步：设置优化器
    # =========================================================================
    # 根据配置选择优化器：
    # - 8-bit AdamW: bitsandbytes 库提供，将优化器状态量化为 8-bit
    #   可减少约 50% 的优化器显存占用
    # - adamw: 标准 AdamW 优化器
    #
    # 注意：只有 LoRA 参数是可训练的
    # 对于 LyCORIS，参数存储在 _lycoris 属性中，需要单独收集

    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # 如果使用 LyCORIS，需要额外收集 _lycoris 的参数
    unwrapped_for_params = accelerator.unwrap_model(transformer)
    if hasattr(unwrapped_for_params, "_lycoris") and unwrapped_for_params._lycoris is not None:
        lycoris_params = list(unwrapped_for_params._lycoris.parameters())
        params_to_optimize = lycoris_params
        logger.info(f"Using LyCORIS parameters: {len(lycoris_params)} tensors")
    
    # 缩放学习率
    # 如果使用多 GPU 或大 batch size，需要相应调整学习率
    learning_rate = config.learning_rate
    if config.scale_lr:
        learning_rate = (
            learning_rate
            * config.train_batch_size
            * config.gradient_accumulation_steps
            * accelerator.num_processes
        )
    
    optimizer = create_optimizer(
        optimizer_type=config.optimizer,
        params=params_to_optimize,
        learning_rate=learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )
    
    logger.info(f"Optimizer: {config.optimizer}, LR: {learning_rate}")
    
    # =========================================================================
    # 第十一步：设置学习率调度器
    # =========================================================================
    # 学习率调度器控制学习率在训练过程中的变化
    # 常用策略：
    # - cosine_with_restarts: 余弦退火，周期性重启，适合长期训练
    # - linear: 线性衰减，简单直接
    # - polynomial: 多项式衰减
    # - constant_with_warmup: 预热后保持恒定
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )
    
    # =========================================================================
    # 第十二步：准备 Accelerator
    # =========================================================================
    # 将所有组件交给 Accelerator 管理
    # 这一步会自动处理设备放置和分布式包装
    
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # 将其他组件移动到正确的设备
    # 这些组件不需要训练，所以不需要交给 accelerator.prepare
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # =========================================================================
    # 第十三步：初始化 Checkpoint 管理器
    # =========================================================================
    checkpoint_manager = CheckpointManager(
        output_dir=config.output_dir,
        total_limit=config.checkpoints_total_limit,
        save_state=config.save_state,
    )
    
    # =========================================================================
    # 第十四步：恢复训练（如果需要）
    # =========================================================================
    global_step = 0
    starting_epoch = 0
    
    if config.resume_from_checkpoint:
        global_step, starting_epoch = checkpoint_manager.load_checkpoint(
            checkpoint_path=config.resume_from_checkpoint,
            accelerator=accelerator,
            transformer=transformer,
            logger=logger,
        )
    
    # =========================================================================
    # 第十五步：训练循环
    # =========================================================================
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = "
        f"{config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps}"
    )
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    
    # 仅在主进程显示进度条
    progress_bar = tqdm(
        range(global_step, config.max_train_steps),
        disable=not accelerator.is_local_main_process,
        ascii=True,  # Use ASCII characters for Windows compatibility
    )
    progress_bar.set_description("Steps")
    
    # 训练循环
    for epoch in range(starting_epoch, config.num_train_epochs):
        if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
            train_dataloader.batch_sampler.set_epoch(epoch)
        transformer.train()
        
        for step, batch in enumerate(train_dataloader):
            # 使用 accelerator.accumulate 处理梯度累积
            with accelerator.accumulate(transformer):
                # -----------------------------------------------------------------
                # 1. 编码图像到 latent 空间
                # -----------------------------------------------------------------
                # VAE 将图像编码为低维 latent 表示
                # Anima 使用的 latent 空间维度比像素空间小得多（通常 8x 压缩）
                
                # 获取或计算 latent
                if batch.get("latents") is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    # QwenImage VAE expects [B, C, T, H, W] for images too.
                    pixel_values = pixel_values.unsqueeze(2)
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=weight_dtype):
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        scale = getattr(getattr(vae, "config", None), "scaling_factor", None)
                        if scale is not None:
                            latents = latents * scale
                
                # -----------------------------------------------------------------
                # 2. 编码文本提示
                # -----------------------------------------------------------------
                # 使用文本编码器将标签转换为 embedding
                
                if text_encoder is None:
                    raise RuntimeError("Text encoder is required for training but was not loaded.")

                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(accelerator.device)

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=weight_dtype):
                    te_out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                    prompt_embeds = getattr(te_out, "last_hidden_state", te_out[0])
                
                # -----------------------------------------------------------------
                # 3. 添加噪声（前向扩散过程）
                # -----------------------------------------------------------------
                # 扩散模型的训练目标是预测添加的噪声
                # 使用 Flow Matching 调度器（Anima 基于 Cosmos，使用流匹配）

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # 随机采样时间步 (0 to 1 for flow matching)
                # Use continuous timesteps for flow matching training
                u = torch.rand(bsz, device=latents.device, dtype=latents.dtype)
                # Reshape for broadcasting: [B, 1, 1, 1, 1] for 5D latents
                u_expanded = u.view(bsz, 1, 1, 1, 1)

                # Flow matching interpolation: x_t = (1 - t) * x_0 + t * noise
                noisy_latents = (1 - u_expanded) * latents + u_expanded * noise

                # Convert to discrete timesteps for model input
                num_train_timesteps = scheduler.config.num_train_timesteps
                timesteps = (u * num_train_timesteps).long()
                timesteps = timesteps.clamp(0, num_train_timesteps - 1)
                
                # -----------------------------------------------------------------
                # 4. 模型预测
                # -----------------------------------------------------------------
                # Transformer 预测噪声或 velocity（取决于调度器配置）

                padding_mask = batch.get("padding_mask")
                if padding_mask is not None:
                    padding_mask = padding_mask.to(accelerator.device, dtype=weight_dtype)
                elif getattr(getattr(transformer, "config", None), "concat_padding_mask", False):
                    # Cosmos transformer expects padding_mask shape (1, 1, H, W)
                    # It will be expanded internally to match batch size
                    padding_mask = torch.ones(
                        (1, 1, noisy_latents.shape[-2], noisy_latents.shape[-1]),
                        device=noisy_latents.device,
                        dtype=noisy_latents.dtype,
                    )

                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                ).sample
                
                # -----------------------------------------------------------------
                # 5. 计算损失
                # -----------------------------------------------------------------
                # 使用 Min-SNR 加权策略，这是扩散模型训练的最佳实践
                # 它解决了不同时间步损失尺度不平衡的问题
                
                if config.min_snr_gamma is not None:
                    # 计算信噪比
                    snr = compute_snr(scheduler, timesteps)
                    gamma = torch.tensor(config.min_snr_gamma, device=snr.device)
                    snr_weight = torch.clamp(gamma / snr, max=1.0)
                    
                    # 加权损失
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * snr_weight
                    loss = loss.mean()
                else:
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # -----------------------------------------------------------------
                # 6. 反向传播
                # -----------------------------------------------------------------
                accelerator.backward(loss)
                
                # 梯度裁剪，防止梯度爆炸
                if accelerator.sync_gradients and config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(transformer.parameters(), config.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()

            # LR scheduler step should happen once per optimizer step, not per forward pass
            if accelerator.sync_gradients:
                lr_scheduler.step()
            
            # -----------------------------------------------------------------
            # 7. 更新进度
            # -----------------------------------------------------------------
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # 日志记录
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": global_step,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                # -----------------------------------------------------------------
                # 8. 保存 checkpoint
                # -----------------------------------------------------------------
                if config.checkpointing_steps > 0 and global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        checkpoint_manager.save_checkpoint(
                            accelerator=accelerator,
                            transformer=transformer,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            global_step=global_step,
                            epoch=epoch,
                        )
                        logger.info(f"Saved checkpoint at step {global_step}")
            
            # 检查是否达到最大步数
            if global_step >= config.max_train_steps:
                break
        
        # -----------------------------------------------------------------
        # 9. 验证
        # -----------------------------------------------------------------
        if (
            config.validation_prompt is not None
            and (epoch + 1) % config.validation_epochs == 0
            and accelerator.is_main_process
        ):
            logger.info(f"Running validation at epoch {epoch + 1}...")
            run_validation(
                accelerator=accelerator,
                pipeline=pipeline,
                prompt=config.validation_prompt,
                num_images=config.num_validation_images,
                output_dir=config.output_dir,
                epoch=epoch,
                global_step=global_step,
            )

        # -----------------------------------------------------------------
        # 10. 保存 checkpoint（按 epoch）
        # -----------------------------------------------------------------
        if config.checkpointing_epochs > 0 and (epoch + 1) % config.checkpointing_epochs == 0:
            if accelerator.is_main_process:
                checkpoint_manager.save_checkpoint(
                    accelerator=accelerator,
                    transformer=transformer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    global_step=global_step,
                    epoch=epoch,
                )
                logger.info(f"Saved checkpoint at epoch {epoch + 1} (step {global_step})")
        
        # 检查是否达到最大步数
        if global_step >= config.max_train_steps:
            break
    
    # =========================================================================
    # 第十六步：保存最终模型
    # =========================================================================
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info("Saving final model...")
        
        # 解包装模型
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        
        # 保存 LoRA 权重
        lora_output_dir = os.path.join(config.output_dir, "final_lora")
        os.makedirs(lora_output_dir, exist_ok=True)
        
        # 使用 PEFT 保存方法
        if config.lora_type == "lora":
            unwrapped_transformer.save_pretrained(lora_output_dir)
        else:
            # LyCORIS 使用不同的保存方法
            lycoris_path = os.path.join(lora_output_dir, "lycoris_weights.pt")
            lycoris_net = getattr(unwrapped_transformer, "_lycoris", None)
            if lycoris_net is None:
                raise RuntimeError("LoKr selected but LyCORIS network was not initialized on transformer.")
            torch.save(lycoris_net.state_dict(), lycoris_path)
        
        logger.info(f"Model saved to {lora_output_dir}")
        
        # 保存训练配置
        config_path = os.path.join(config.output_dir, "training_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(vars(config), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Config saved to {config_path}")
    
    accelerator.end_training()
    logger.info("Training completed!")


def compute_snr(scheduler, timesteps):
    """
    计算信噪比（Signal-to-Noise Ratio）
    
    这是 Min-SNR 加权策略的核心计算
    用于解决扩散模型训练中不同时间步损失不平衡的问题
    
    公式：SNR = alpha_t^2 / sigma_t^2
    其中 alpha_t 是信号系数，sigma_t 是噪声系数
    
    Args:
        scheduler: 噪声调度器
        timesteps: 时间步张量
    
    Returns:
        snr: 信噪比张量
    """
    if hasattr(scheduler, "alphas_cumprod"):
        alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
        sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod[timesteps]) ** 0.5

        # SNR = alpha^2 / sigma^2
        snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
        return snr

    # FlowMatch scheduler (uses sigmas)
    if hasattr(scheduler, "sigmas") and scheduler.sigmas is not None:
        sigmas = scheduler.sigmas.to(device=timesteps.device, dtype=torch.float32)
        num_sigmas = len(sigmas)

        # For flow matching, compute sigma from continuous timesteps
        # timesteps are discrete indices; convert to sigma values safely
        num_train_timesteps = getattr(scheduler.config, "num_train_timesteps", 1000)

        # Clamp indices to valid range
        step_indices = timesteps.clamp(0, num_sigmas - 1).long()
        sigma = sigmas[step_indices].to(timesteps.device)

        # Avoid division by zero
        sigma = sigma.clamp(min=1e-8)
        snr = ((1.0 - sigma) / sigma) ** 2
        return snr

    return torch.ones_like(timesteps, dtype=torch.float32)


def run_validation(
    accelerator,
    pipeline,
    prompt: str,
    num_images: int,
    output_dir: str,
    epoch: int,
    global_step: int,
):
    """
    运行验证，生成样本图像
    
    Args:
        accelerator: Accelerator 实例
        pipeline: 扩散模型 pipeline
        prompt: 验证提示词
        num_images: 生成图像数量
        output_dir: 输出目录
        epoch: 当前 epoch
        global_step: 当前全局步数
    """
    # 切换到评估模式
    pipeline.transformer.eval()

    validation_dir = os.path.join(output_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(42)

    images = []
    try:
        for i in range(num_images):
            with torch.no_grad():
                image = pipeline(
                    prompt=prompt,
                    num_inference_steps=30,
                    generator=generator,
                ).images[0]

            # 保存图像
            image_path = os.path.join(
                validation_dir,
                f"epoch{epoch}_step{global_step}_sample{i}.png"
            )
            image.save(image_path)
            images.append(image)

        # 记录到 WandB
        if accelerator.is_main_process and accelerator.trackers is not None:
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    import wandb
                    tracker.log(
                        {
                            f"validation_epoch_{epoch}": [
                                wandb.Image(img, caption=f"Sample {i}: {prompt}")
                                for i, img in enumerate(images)
                            ]
                        },
                        step=global_step,
                    )
    finally:
        # 切换回训练模式（确保即使出错也能恢复）
        pipeline.transformer.train()


if __name__ == "__main__":
    main()
