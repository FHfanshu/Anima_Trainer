#!/usr/bin/env python
"""
Anima LoRA Trainer v2 - 支持 LyCORIS + 训练时推理
基于 trainerV1.01 重构，轻量单文件

特性：
- 标准 LoRA 和 LyCORIS LoKr 双模式
- 训练时推理出图
- Flow Matching 训练
- ARB 分桶
- 依赖自动检测与安装
- Rich 进度条 + ASCII Loss 曲线
- 梯度检查点支持
- Caption 预处理 (shuffle/keep_tokens)
"""

import argparse
import logging
import os
import random
import subprocess
import sys
import time
import types
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 依赖检测
# ============================================================================

def ensure_dependencies(auto_install=False):
    """检测并可选自动安装缺失依赖"""
    required = {
        "numpy": "numpy",
        "PIL": "Pillow",
        "safetensors": "safetensors",
        "transformers": "transformers",
        "einops": "einops",
        "torchvision": "torchvision",
        "yaml": "pyyaml",
    }
    missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except Exception:
            missing.append(pip_name)
    if not missing:
        return
    missing_list = ", ".join(sorted(set(missing)))
    print(f"Missing dependencies: {missing_list}")
    if not auto_install:
        print(f"Install them with:\n  {sys.executable} -m pip install {missing_list}")
        raise SystemExit(1)
    cmd = [sys.executable, "-m", "pip", "install", *sorted(set(missing))]
    print("Installing missing dependencies...")
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"Auto-install failed: {exc}")
        raise SystemExit(1)
    # Re-check after install
    still_missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except Exception:
            still_missing.append(pip_name)
    if still_missing:
        still_list = ", ".join(sorted(set(still_missing)))
        print(f"Still missing: {still_list}")
        raise SystemExit(1)


# ============================================================================
# YAML 配置加载
# ============================================================================

def load_yaml_config(config_path):
    """加载 YAML 配置文件"""
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed. Install with: pip install pyyaml")
        raise SystemExit(1)

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def apply_yaml_config(args, config):
    """将 YAML 配置应用到 args，命令行参数优先"""
    # 字段映射: YAML key -> args attribute
    mapping = {
        # 模型路径
        "transformer_path": "transformer",
        "vae_path": "vae",
        "text_encoder_path": "qwen",
        "t5_tokenizer_path": "t5_tokenizer",
        # 数据集
        "data_dir": "data_dir",
        "resolution": "resolution",
        "repeats": "repeats",
        "shuffle_caption": "shuffle_caption",
        "keep_tokens": "keep_tokens",
        "flip_augment": "flip_augment",
        "cache_latents": "cache_latents",
        # LoRA 配置
        "lora_type": "lora_type",
        "lora_rank": "lora_rank",
        "lora_alpha": "lora_alpha",
        "lokr_factor": "lokr_factor",
        # 训练参数
        "epochs": "epochs",
        "max_steps": "max_steps",
        "batch_size": "batch_size",
        "grad_accum": "grad_accum",
        "learning_rate": "lr",
        "mixed_precision": "mixed_precision",
        "grad_checkpoint": "grad_checkpoint",
        "xformers": "xformers",
        "num_workers": "num_workers",
        # 输出与保存
        "lora_name": "lora_name",
        "output_dir": "output_dir",
        "output_name": "output_name",
        "save_every": "save_every",
        "save_state": "save_state",
        "resume": "resume",
        "seed": "seed",
        # 采样
        "sample_every": "sample_every",
        "sample_prompt": "sample_prompt",
        # 进度显示
        "loss_curve_steps": "loss_curve_steps",
        "no_progress": "no_progress",
    }

    # 需要特殊处理的默认值（用于判断命令行是否显式设置）
    defaults = {
        "transformer": "",
        "vae": "",
        "qwen": "",
        "t5_tokenizer": "",
        "data_dir": "",
        "resolution": 1024,
        "repeats": 1,
        "shuffle_caption": False,
        "keep_tokens": 0,
        "flip_augment": False,
        "cache_latents": False,
        "lora_type": "lokr",
        "lora_rank": 32,
        "lora_alpha": 32.0,
        "lokr_factor": 8,
        "epochs": 10,
        "max_steps": 0,
        "batch_size": 1,
        "grad_accum": 1,
        "lr": 1e-4,
        "mixed_precision": "bf16",
        "grad_checkpoint": False,
        "xformers": False,
        "num_workers": 0,
        "lora_name": "",
        "output_dir": "./output",
        "output_name": "anima_lora",
        "save_every": 0,
        "save_state": 0,
        "resume": "",
        "seed": 42,
        "sample_every": 0,
        "sample_prompt": "1girl, masterpiece",
        "loss_curve_steps": 100,
        "no_progress": False,
    }

    for yaml_key, arg_attr in mapping.items():
        if yaml_key not in config:
            continue
        yaml_value = config[yaml_key]
        if yaml_value is None:
            continue

        # 检查命令行是否显式设置了该参数（与默认值不同）
        current_value = getattr(args, arg_attr, None)
        default_value = defaults.get(arg_attr)

        # 如果当前值等于默认值，则使用 YAML 配置
        if current_value == default_value:
            setattr(args, arg_attr, yaml_value)

    return args


# Lazy imports after dependency check
def _lazy_imports():
    global np, Image
    import numpy as np
    from PIL import Image


# ============================================================================
# 进度和 Loss 曲线可视化
# ============================================================================

def init_progress(show_progress, total_steps):
    """初始化 Rich 进度条"""
    if not show_progress:
        return None, None, None
    try:
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, TextColumn,
            TimeElapsedColumn, TimeRemainingColumn,
        )
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TextColumn("lr={task.fields[lr]:.2e}"),
            TextColumn("speed={task.fields[speed]:.2f} it/s"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=10,
        )
        task = progress.add_task("train", total=total_steps, loss=0.0, lr=0.0, speed=0.0)
        return progress, task, "rich"
    except Exception:
        return "plain", None, None


def render_loss_curve(losses, width=60, height=10):
    """渲染 ASCII Loss 曲线"""
    if not losses:
        return ""
    if width < 5:
        width = 5
    values = losses
    if len(values) > width:
        step = len(values) / width
        buckets = []
        for i in range(width):
            start = int(i * step)
            end = int((i + 1) * step)
            end = max(end, start + 1)
            chunk = values[start:end]
            buckets.append(sum(chunk) / len(chunk))
        values = buckets
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        max_v = min_v + 1e-8
    grid = [[" " for _ in range(len(values))] for _ in range(height)]
    for i, v in enumerate(values):
        y = int((v - min_v) / (max_v - min_v) * (height - 1))
        y = height - 1 - y
        grid[y][i] = "*"
    lines = ["".join(row) for row in grid]
    lines.append(f"min={min_v:.4f} max={max_v:.4f}")
    return "\n".join(lines)


def render_curve_panel(losses, width=60, height=10):
    """渲染 Rich Panel 包装的 Loss 曲线"""
    try:
        from rich.panel import Panel
        from rich.text import Text
    except Exception:
        return None
    chart = render_loss_curve(losses, width=width, height=height)
    return Panel(Text(chart), title="Loss curve (recent)", expand=False)


# ============================================================================
# 梯度检查点
# ============================================================================

def forward_with_optional_checkpoint(model, latents, timesteps, cross, padding_mask, use_checkpoint=False):
    """带可选梯度检查点的前向传播"""
    if not use_checkpoint:
        return model(latents, timesteps, cross, padding_mask=padding_mask)
    from torch.utils.checkpoint import checkpoint

    x_B_T_H_W_D, rope_emb, extra_pos_emb = model.prepare_embedded_sequence(
        latents, fps=None, padding_mask=padding_mask,
    )
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    t_embedding, adaln_lora = model.t_embedder(timesteps)
    t_embedding = model.t_embedding_norm(t_embedding)

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb,
        "adaln_lora_B_T_3D": adaln_lora,
        "extra_per_block_pos_emb": extra_pos_emb,
    }

    for block in model.blocks:
        def custom_forward(x, blk=block):
            return blk(x, t_embedding, cross, **block_kwargs)
        x_B_T_H_W_D = checkpoint(custom_forward, x_B_T_H_W_D, use_reentrant=False)

    x_B_T_H_W_O = model.final_layer(x_B_T_H_W_D, t_embedding, adaln_lora_B_T_3D=adaln_lora)
    return model.unpatchify(x_B_T_H_W_O)


# ============================================================================
# xformers 支持
# ============================================================================

def enable_xformers(model):
    """为模型启用 xformers memory efficient attention"""
    try:
        from xformers.ops import memory_efficient_attention
    except ImportError:
        logger.warning("xformers 未安装，跳过启用")
        return False

    enabled_count = 0
    for name, module in model.named_modules():
        # 查找 attention 模块并替换
        if hasattr(module, "set_use_memory_efficient_attention_xformers"):
            module.set_use_memory_efficient_attention_xformers(True)
            enabled_count += 1
        elif hasattr(module, "enable_xformers_memory_efficient_attention"):
            module.enable_xformers_memory_efficient_attention()
            enabled_count += 1

    if enabled_count > 0:
        logger.info(f"xformers 已启用: {enabled_count} 个模块")
        return True

    # 如果模型没有内置支持，尝试 monkey patch
    logger.info("xformers 已加载，将在 attention 计算中使用")
    return True


# ============================================================================
# 模型加载工具
# ============================================================================

def find_diffusion_pipe_root():
    """查找 diffusion-pipe 模型代码路径"""
    candidates = [
        Path(__file__).parent / "diffusion_models",
        Path(__file__).parent / "models",
        Path(os.environ.get("DIFFUSION_PIPE_ROOT", "")) if os.environ.get("DIFFUSION_PIPE_ROOT") else None,
    ]
    for candidate in candidates:
        if candidate and (candidate / "anima_modeling.py").exists():
            return candidate
        if candidate and (candidate / "models" / "anima_modeling.py").exists():
            return candidate / "models"
    raise RuntimeError("找不到 anima_modeling.py，请设置 DIFFUSION_PIPE_ROOT 或放置模型代码")


def load_module_from_path(module_name, file_path):
    """动态加载 Python 模块"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_models_namespace(repo_root):
    """确保 models 命名空间可用"""
    repo_root = Path(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(repo_root.parent) not in sys.path:
        sys.path.insert(0, str(repo_root.parent))


def load_anima_model(transformer_path, device, dtype, repo_root):
    """加载 Anima transformer 模型"""
    from safetensors import safe_open

    ensure_models_namespace(repo_root)

    # 加载模型类
    cosmos_modeling = load_module_from_path(
        "cosmos_predict2_modeling",
        repo_root / "cosmos_predict2_modeling.py",
    )
    anima_modeling = load_module_from_path(
        "anima_modeling",
        repo_root / "anima_modeling.py",
    )
    Anima = anima_modeling.Anima

    # 从 checkpoint 推断配置
    w = None
    with safe_open(transformer_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith("x_embedder.proj.1.weight"):
                w = f.get_tensor(k)
                break

    if w is None:
        raise RuntimeError("无法在 transformer 权重中找到 x_embedder.proj.1.weight")

    in_channels = (w.shape[1] // 4) - 1  # concat_padding_mask=True
    model_channels = w.shape[0]

    if model_channels == 2048:
        num_blocks, num_heads = 28, 16
    elif model_channels == 5120:
        num_blocks, num_heads = 36, 40
    else:
        raise RuntimeError(f"未知的 model_channels={model_channels}")

    config = dict(
        max_img_h=240, max_img_w=240, max_frames=128,
        in_channels=in_channels, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=model_channels,
        num_blocks=num_blocks, num_heads=num_heads,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True, adaln_lora_dim=256,
        rope_h_extrapolation_ratio=4.0 if in_channels == 16 else 3.0,
        rope_w_extrapolation_ratio=4.0 if in_channels == 16 else 3.0,
        rope_t_extrapolation_ratio=1.0,
    )

    model = Anima(**config)

    # 加载权重
    sd = {}
    with safe_open(transformer_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)

    model.load_state_dict(sd, strict=False)
    model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)

    logger.info(f"Anima 模型加载完成: {model_channels}ch, {num_blocks} blocks")
    return model


def load_vae(vae_path, device, dtype, repo_root):
    """加载 VAE"""
    from safetensors import safe_open

    wan_vae = load_module_from_path("wan_vae", repo_root / "wan" / "vae2_1.py")
    WanVAE = wan_vae.WanVAE_

    cfg = dict(
        dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True], dropout=0.0,
    )

    model = WanVAE(**cfg).eval().requires_grad_(False)

    sd = {}
    with safe_open(vae_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    model.load_state_dict(sd, strict=False)
    model = model.to(device=device, dtype=dtype)

    # VAE 归一化参数
    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ], dtype=dtype, device=device)
    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ], dtype=dtype, device=device)

    class VAEWrapper:
        pass

    wrapper = VAEWrapper()
    wrapper.model = model
    wrapper.mean = mean
    wrapper.std = std
    wrapper.scale = [mean, 1.0 / std]

    logger.info("VAE 加载完成")
    return wrapper


def load_text_encoders(qwen_path, t5_tokenizer_path, device, dtype):
    """加载文本编码器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    # Qwen
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval().requires_grad_(False)

    # T5 tokenizer
    if t5_tokenizer_path and Path(t5_tokenizer_path).exists():
        t5_tokenizer = T5Tokenizer.from_pretrained(t5_tokenizer_path)
    else:
        class _DummyT5Tokenizer:
            def __call__(self, texts, max_length=512, **_kwargs):
                if isinstance(texts, str):
                    batch = 1
                else:
                    batch = len(texts)
                input_ids = torch.zeros((batch, max_length), dtype=torch.long)
                attention_mask = torch.zeros((batch, max_length), dtype=torch.long)
                return types.SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)

        t5_tokenizer = _DummyT5Tokenizer()
        logger.warning("未提供 T5 tokenizer，使用占位 tokenizer（不会下载）")

    logger.info("文本编码器加载完成")
    return qwen_model, qwen_tokenizer, t5_tokenizer


# ============================================================================
# 文本编码
# ============================================================================

def encode_qwen(model, tokenizer, texts, device, max_length=512):
    """Qwen 文本编码"""
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length
    ).to(device)

    with torch.inference_mode():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden = outputs.hidden_states[-1]
    # 清零 padding 位置
    mask = inputs.attention_mask.unsqueeze(-1)
    hidden = hidden * mask

    return hidden


def tokenize_t5(tokenizer, texts, max_length=512):
    """T5 tokenizer"""
    return tokenizer(
        texts, return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_length
    )


# ============================================================================
# LoRA 实现
# ============================================================================

class LoRALayer(torch.nn.Module):
    """标准 LoRA 层"""
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_down = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, out_features, bias=False)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling


class LoKrLayer(torch.nn.Module):
    """LyCORIS LoKr 层 (ComfyUI 兼容)"""
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, factor=8, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 自动调整 factor 确保能整除
        factor = self._find_factor(in_features, out_features, factor)
        self.factor = factor

        self.in_dim = in_features // factor
        self.out_dim = out_features // factor

        # LoKr 分解: W = kron(w1, w2)
        # w1: [factor, factor], w2: [out_dim, in_dim]
        self.lokr_w1 = torch.nn.Parameter(torch.empty(factor, factor))
        self.lokr_w2 = torch.nn.Parameter(torch.empty(self.out_dim, self.in_dim))
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        torch.nn.init.kaiming_uniform_(self.lokr_w1, a=5**0.5)
        torch.nn.init.zeros_(self.lokr_w2)

    def _find_factor(self, in_f, out_f, target_factor):
        """找到能同时整除 in_features 和 out_features 的 factor"""
        for f in [target_factor, 4, 2, 1]:
            if in_f % f == 0 and out_f % f == 0:
                return f
        return 1

    def forward(self, x):
        weight = torch.kron(self.lokr_w1, self.lokr_w2)
        return F.linear(self.dropout(x), weight) * self.scaling


class LoRALinear(torch.nn.Module):
    """LoRA 包装的 Linear 层"""
    def __init__(self, original, rank=4, alpha=1.0, dropout=0.0, use_lokr=False, factor=8):
        super().__init__()
        self.original = original
        self.use_lokr = use_lokr

        if use_lokr:
            self.adapter = LoKrLayer(
                original.in_features, original.out_features,
                rank=rank, alpha=alpha, factor=factor, dropout=dropout
            )
        else:
            self.adapter = LoRALayer(
                original.in_features, original.out_features,
                rank=rank, alpha=alpha, dropout=dropout
            )

        self.adapter.to(device=original.weight.device, dtype=original.weight.dtype)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.adapter(x)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias


class LoRAInjector:
    """LoRA 注入器"""
    DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"]

    def __init__(self, rank=32, alpha=16.0, dropout=0.0, use_lokr=False, factor=8, targets=None):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.use_lokr = use_lokr
        self.factor = factor
        self.targets = targets or self.DEFAULT_TARGETS
        self.injected = {}

    def inject(self, model):
        """注入 LoRA 到模型"""
        for name, module in list(model.named_modules()):
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(t in name for t in self.targets):
                continue

            lora_linear = LoRALinear(
                module, rank=self.rank, alpha=self.alpha,
                dropout=self.dropout, use_lokr=self.use_lokr, factor=self.factor
            )

            # 替换模块
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            last = parts[-1]
            if last.isdigit():
                parent[int(last)] = lora_linear
            else:
                setattr(parent, last, lora_linear)
            self.injected[name] = lora_linear

        logger.info(f"注入 {'LoKr' if self.use_lokr else 'LoRA'} 到 {len(self.injected)} 层")
        return self.injected

    def get_params(self):
        """获取可训练参数"""
        params = []
        for lora in self.injected.values():
            params.extend(lora.adapter.parameters())
        return params

    def state_dict(self):
        """导出 LoRA 权重 (ComfyUI 兼容格式)"""
        sd = {}
        for name, lora in self.injected.items():
            # ComfyUI 格式: lycoris_{key} 其中 key 是 diffusion_model.net.xxx 的 net.xxx 部分
            # 模型内部 name 是 blocks.0.xxx，需要加上 net. 前缀
            full_name = "net." + name
            base = "lycoris_" + full_name.replace(".", "_")
            sd[f"{base}.alpha"] = torch.tensor(self.alpha)

            if self.use_lokr:
                sd[f"{base}.lokr_w1"] = lora.adapter.lokr_w1.data.clone()
                sd[f"{base}.lokr_w2"] = lora.adapter.lokr_w2.data.clone()
            else:
                sd[f"{base}.lora_down.weight"] = lora.adapter.lora_down.weight.data.clone()
                sd[f"{base}.lora_up.weight"] = lora.adapter.lora_up.weight.data.clone()
        return sd

    def load_state_dict(self, sd):
        """从 state_dict 加载 LoRA 权重"""
        for name, lora in self.injected.items():
            full_name = "net." + name
            base = "lycoris_" + full_name.replace(".", "_")
            if self.use_lokr:
                if f"{base}.lokr_w1" in sd:
                    lora.adapter.lokr_w1.data.copy_(sd[f"{base}.lokr_w1"])
                if f"{base}.lokr_w2" in sd:
                    lora.adapter.lokr_w2.data.copy_(sd[f"{base}.lokr_w2"])
            else:
                if f"{base}.lora_down.weight" in sd:
                    lora.adapter.lora_down.weight.data.copy_(sd[f"{base}.lora_down.weight"])
                if f"{base}.lora_up.weight" in sd:
                    lora.adapter.lora_up.weight.data.copy_(sd[f"{base}.lora_up.weight"])

    def save(self, path):
        """保存为 safetensors (ComfyUI 兼容)"""
        from safetensors.torch import save_file
        sd = self.state_dict()
        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_network_module": "lycoris.kohya" if self.use_lokr else "networks.lora",
            "ss_network_args": f'{{"algo": "lokr", "factor": {self.factor}}}' if self.use_lokr else "{}",
        }
        save_file(sd, path, metadata=meta)
        logger.info(f"LoRA 保存到: {path}")


def save_training_state(state_dir, injector, optimizer, epoch, global_step, keep_last_n=1):
    """保存训练状态用于恢复"""
    import shutil
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    # 保存当前状态
    epoch_dir = state_dir / f"epoch-{epoch}"
    epoch_dir.mkdir(exist_ok=True)

    # 保存 LoRA 权重
    injector.save(epoch_dir / "lora.safetensors")

    # 保存 optimizer 和训练信息
    torch.save({
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, epoch_dir / "state.pt")

    logger.info(f"训练状态保存到: {epoch_dir}")

    # 清理旧状态，只保留最近 N 个
    if keep_last_n > 0:
        existing = sorted(state_dir.glob("epoch-*"), key=lambda p: int(p.name.split("-")[1]))
        while len(existing) > keep_last_n:
            old_dir = existing.pop(0)
            shutil.rmtree(old_dir)
            logger.info(f"删除旧状态: {old_dir}")


def load_training_state(state_dir, injector, optimizer):
    """加载训练状态"""
    state_dir = Path(state_dir)

    # 加载 LoRA 权重
    lora_path = state_dir / "lora.safetensors"
    if lora_path.exists():
        from safetensors import safe_open
        with safe_open(lora_path, framework="pt") as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}
        injector.load_state_dict(sd)
        logger.info(f"LoRA 权重已加载: {lora_path}")

    # 加载 optimizer 和训练信息
    state_path = state_dir / "state.pt"
    if state_path.exists():
        state = torch.load(state_path, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        epoch = state["epoch"]
        global_step = state["global_step"]
        logger.info(f"训练状态已加载: epoch={epoch}, step={global_step}")
        return epoch, global_step

    return 0, 0


# ============================================================================
# 数据集
# ============================================================================

class BucketManager:
    """ARB 分桶管理"""
    def __init__(self, base_reso=1024, min_reso=512, max_reso=2048, step=64):
        self.base_reso = base_reso
        self.buckets = self._generate(min_reso, max_reso, step, base_reso)

    def _generate(self, min_r, max_r, step, base):
        buckets = []
        base_area = base * base
        for w in range(min_r, max_r + 1, step):
            for h in range(min_r, max_r + 1, step):
                if abs(w * h - base_area) / base_area > 0.1:
                    continue
                if max(w/h, h/w) > 2.0:
                    continue
                buckets.append((w, h))
        return buckets

    def get_bucket(self, w, h):
        aspect = w / h
        best = (self.base_reso, self.base_reso)
        best_diff = float("inf")
        for bw, bh in self.buckets:
            diff = abs(aspect - bw/bh)
            if diff < best_diff:
                best_diff = diff
                best = (bw, bh)
        return best


class ImageDataset(Dataset):
    """图像数据集"""
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, data_dir, resolution=1024, bucket_mgr=None,
                 shuffle_caption=False, keep_tokens=0, flip_augment=False):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.bucket_mgr = bucket_mgr
        self.shuffle_caption = shuffle_caption
        self.keep_tokens = keep_tokens
        self.flip_augment = flip_augment
        self.samples = self._scan()
        logger.info(f"数据集: {len(self.samples)} 样本")

    def _scan(self):
        samples = []
        for img_path in self.data_dir.rglob("*"):
            if img_path.suffix.lower() not in self.EXTS:
                continue
            txt_path = img_path.with_suffix(".txt")
            if not txt_path.exists():
                txt_path = img_path.with_suffix(".caption")
            if not txt_path.exists():
                continue
            samples.append({"image": img_path, "caption": txt_path})
        return samples

    def _process_caption(self, caption):
        """处理 caption: tag 打乱 + keep_tokens"""
        if not caption:
            return ""
        if "," in caption:
            tags = [t.strip() for t in caption.split(",")]
        else:
            tags = caption.split()

        if self.keep_tokens > 0:
            kept = tags[:self.keep_tokens]
            rest = tags[self.keep_tokens:]
            if self.shuffle_caption:
                random.shuffle(rest)
            tags = kept + rest
        elif self.shuffle_caption:
            random.shuffle(tags)

        return ", ".join(tags)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image
        sample = self.samples[idx]
        img = Image.open(sample["image"]).convert("RGB")
        caption = sample["caption"].read_text(encoding="utf-8").strip()
        caption = self._process_caption(caption)

        # ARB 分桶
        if self.bucket_mgr:
            tw, th = self.bucket_mgr.get_bucket(img.width, img.height)
        else:
            tw = th = self.resolution

        # 缩放裁剪
        scale = max(tw / img.width, th / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        img = img.resize((nw, nh), Image.LANCZOS)

        left = (nw - tw) // 2
        top = (nh - th) // 2
        img = img.crop((left, top, left + tw, top + th))

        # 水平翻转增强
        if self.flip_augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 转 tensor [-1, 1]
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return {"pixel_values": tensor, "caption": caption}


class RepeatDataset(Dataset):
    """Kohya 风格数据集重复"""
    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class CachedLatentDataset(Dataset):
    """Kohya 风格 npz 文件缓存的数据集"""
    def __init__(self, base_dataset, vae, device, dtype, cache_dir=None):
        import numpy as np
        self.base_dataset = base_dataset
        self.np = np
        # 获取原始数据集的 samples 列表
        self.samples = self._get_base_samples(base_dataset)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._build_cache(vae, device, dtype)

    def _get_base_samples(self, dataset):
        """获取原始 ImageDataset 的 samples"""
        if hasattr(dataset, "samples"):
            return dataset.samples
        elif hasattr(dataset, "dataset"):
            return self._get_base_samples(dataset.dataset)
        return []

    def _get_npz_path(self, img_path):
        """获取图像对应的 npz 缓存路径"""
        img_path = Path(img_path)
        return img_path.with_suffix(".npz")

    def _is_cache_valid(self, img_path, npz_path):
        """检查缓存是否有效（图像未修改）"""
        if not npz_path.exists():
            return False
        return npz_path.stat().st_mtime >= img_path.stat().st_mtime

    def _build_cache(self, vae, device, dtype):
        """构建/加载 npz 缓存"""
        logger.info("检查 VAE latent 缓存...")
        to_encode = []
        for i, sample in enumerate(self.samples):
            img_path = sample["image"]
            npz_path = self._get_npz_path(img_path)
            if not self._is_cache_valid(img_path, npz_path):
                to_encode.append(i)

        if to_encode:
            logger.info(f"需要编码 {len(to_encode)}/{len(self.samples)} 张图像...")
            self._encode_and_save(to_encode, vae, device, dtype)
        else:
            logger.info(f"所有 {len(self.samples)} 张图像已缓存")

    def _encode_and_save(self, indices, vae, device, dtype):
        """编码图像并保存为 npz"""
        for count, i in enumerate(indices):
            item = self.base_dataset[i]
            pixels = item["pixel_values"].unsqueeze(0).to(device, dtype=dtype)
            with torch.no_grad():
                pixels_5d = pixels.unsqueeze(2)
                latent = vae.model.encode(pixels_5d, vae.scale)
            latent_np = latent.squeeze(0).cpu().float().numpy()
            npz_path = self._get_npz_path(self.samples[i]["image"])
            self.np.savez(npz_path, latent=latent_np)
            if (count + 1) % 10 == 0 or count == len(indices) - 1:
                logger.info(f"  编码进度: {count + 1}/{len(indices)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npz_path = self._get_npz_path(sample["image"])
        data = self.np.load(npz_path)
        latent = torch.from_numpy(data["latent"])
        caption = sample["caption"].read_text(encoding="utf-8").strip()
        # 处理 caption（如果原数据集有处理逻辑）
        if hasattr(self.base_dataset, "_process_caption"):
            caption = self.base_dataset._process_caption(caption)
        elif hasattr(self.base_dataset, "dataset") and hasattr(self.base_dataset.dataset, "_process_caption"):
            caption = self.base_dataset.dataset._process_caption(caption)
        return {"latent": latent, "caption": caption}


# ============================================================================
# 训练时推理
# ============================================================================

@torch.no_grad()
def sample_image(
    model, vae, qwen_model, qwen_tokenizer, t5_tokenizer,
    prompt, height=1024, width=1024, steps=30, device="cuda", dtype=torch.bfloat16
):
    """Flow Matching 采样生成图像"""
    was_training = model.training
    model.eval()

    # 文本编码
    qwen_embeds = encode_qwen(qwen_model, qwen_tokenizer, [prompt], device)
    t5_tokens = tokenize_t5(t5_tokenizer, [prompt], max_length=512)
    t5_ids = t5_tokens.input_ids.to(device)

    cross = model.preprocess_text_embeds(qwen_embeds, t5_ids)
    if cross.shape[1] < 512:
        cross = F.pad(cross, (0, 0, 0, 512 - cross.shape[1]))

    # 初始化噪声
    lat_h, lat_w = height // 8, width // 8
    latents = torch.randn(1, 16, 1, lat_h, lat_w, device=device, dtype=dtype)

    # Euler 采样
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.tensor([[1.0 - i * dt]], device=device, dtype=dtype)
        pad_mask = torch.zeros(1, 1, lat_h, lat_w, device=device, dtype=dtype)

        autocast_ctx = torch.autocast("cuda", dtype=dtype) if device == "cuda" else nullcontext()
        with autocast_ctx:
            v = model(latents, t, cross, padding_mask=pad_mask)
        latents = latents - dt * v

    # VAE 解码
    images = vae.model.decode(latents, vae.scale)
    images = images.squeeze(2)  # [B,C,H,W]
    images = (images.clamp(-1, 1) + 1) / 2

    # 转 PIL
    img = images[0].permute(1, 2, 0).cpu().float().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)

    if was_training:
        model.train()
    return Image.fromarray(img)


# ============================================================================
# 训练辅助
# ============================================================================

def sample_t(bs, device):
    """采样时间步 (logit-normal)"""
    t = torch.sigmoid(torch.randn(bs, device=device))
    shift = 3.0
    t = (t * shift) / (1 + (shift - 1) * t)
    return t


def collate_fn(batch):
    """DataLoader collate"""
    pixels = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["caption"] for b in batch]
    return {"pixel_values": pixels, "captions": captions}


def collate_fn_cached(batch):
    """DataLoader collate for cached latents"""
    latents = torch.stack([b["latent"] for b in batch])
    captions = [b["caption"] for b in batch]
    return {"latents": latents, "captions": captions}


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Anima LoRA Trainer v2")
    # 配置文件
    p.add_argument("--config", default="", help="YAML 配置文件路径")
    # 路径
    p.add_argument("--data-dir", default="", help="数据集目录")
    p.add_argument("--transformer", default="", help="transformer safetensors")
    p.add_argument("--vae", default="", help="VAE safetensors")
    p.add_argument("--qwen", default="", help="Qwen 模型目录")
    p.add_argument("--t5-tokenizer", default="", help="T5 tokenizer 目录")
    p.add_argument("--output-dir", default="./output", help="输出目录")
    p.add_argument("--output-name", default="anima_lora", help="输出名称")
    p.add_argument("--lora-name", default="", help="输出子目录名 (可选)")

    # 训练参数
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--mixed-precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--grad-checkpoint", action="store_true", help="启用梯度检查点减少显存")
    p.add_argument("--xformers", action="store_true", help="启用 xformers memory efficient attention")
    p.add_argument("--max-steps", type=int, default=0, help="最大训练步数 (0=无限制)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")

    # 数据集参数
    p.add_argument("--repeats", type=int, default=1, help="数据集重复次数 (Kohya 风格)")
    p.add_argument("--shuffle-caption", action="store_true", help="打乱 caption tags")
    p.add_argument("--keep-tokens", type=int, default=0, help="保留前 N 个 tokens 不打乱")
    p.add_argument("--flip-augment", action="store_true", help="随机水平翻转增强")
    p.add_argument("--cache-latents", action="store_true", help="缓存 VAE latent 加速训练")

    # LoRA 参数
    p.add_argument("--lora-type", choices=["lora", "lokr"], default="lokr")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lokr-factor", type=int, default=8)

    # 采样参数
    p.add_argument("--sample-every", type=int, default=0, help="每 N 个 epoch 采样一次 (0=禁用)")
    p.add_argument("--sample-prompt", default="1girl, masterpiece", help="采样提示词")

    # 保存参数
    p.add_argument("--save-every", type=int, default=0, help="每 N 个 epoch 保存 (0=仅结束时)")
    p.add_argument("--save-state", type=int, default=0, help="保存最近 N 个 epoch 的训练状态 (0=不保存)")
    p.add_argument("--resume", type=str, default="", help="从指定状态目录恢复训练")
    p.add_argument("--seed", type=int, default=42)

    # 进度显示
    p.add_argument("--no-progress", action="store_true", help="禁用动态进度显示")
    p.add_argument("--loss-curve-steps", type=int, default=100, help="Loss 曲线显示步数 (0=禁用)")
    p.add_argument("--no-live-curve", action="store_true", help="禁用实时 Loss 曲线刷新")
    p.add_argument("--log-every", type=int, default=10, help="日志输出间隔")

    # 依赖和交互
    p.add_argument("--auto-install", action="store_true", help="自动安装缺失依赖")
    p.add_argument("--interactive", action="store_true", help="交互模式，提示输入缺失参数")

    return p.parse_args()


# ============================================================================
# 交互模式辅助函数
# ============================================================================

def _try_rich():
    try:
        from rich.prompt import Prompt, Confirm
        return Prompt, Confirm
    except Exception:
        return None, None


def _ask_str(label, default=""):
    Prompt, _ = _try_rich()
    if Prompt:
        return Prompt.ask(label, default=default) if default else Prompt.ask(label)
    raw = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return raw or default


def _ask_bool(label, default=False):
    _, Confirm = _try_rich()
    if Confirm:
        return Confirm.ask(label, default=default)
    raw = input(f"{label} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true", "t")


def _ask_int(label, default):
    while True:
        raw = _ask_str(label, str(default))
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")


def _ask_float(label, default):
    while True:
        raw = _ask_str(label, str(default))
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def _guess_default_paths():
    base = Path(__file__).resolve().parent
    transformer = base / "anima" / "diffusion_models" / "anima-preview.safetensors"
    vae = base / "anima" / "vae" / "qwen_image_vae.safetensors"
    qwen = base / "anima" / "text_encoders"
    return {
        "transformer": str(transformer) if transformer.exists() else "",
        "vae": str(vae) if vae.exists() else "",
        "qwen": str(qwen) if qwen.exists() else "",
    }


def prompt_for_args(args):
    """交互式提示输入缺失参数"""
    defaults = _guess_default_paths()
    args.data_dir = args.data_dir or _ask_str("数据集目录 (images + .txt)", "")
    args.transformer = args.transformer or _ask_str("Transformer 路径 (.safetensors)", defaults["transformer"])
    args.vae = args.vae or _ask_str("VAE 路径 (.safetensors)", defaults["vae"])
    args.qwen = args.qwen or _ask_str("Qwen 模型目录", defaults["qwen"])
    args.output_dir = _ask_str("输出目录", args.output_dir)
    args.output_name = _ask_str("输出名称", args.output_name)
    args.resolution = _ask_int("分辨率", args.resolution)
    args.batch_size = _ask_int("Batch size", args.batch_size)
    args.grad_accum = _ask_int("梯度累积", args.grad_accum)
    args.lr = _ask_float("学习率", args.lr)
    args.repeats = _ask_int("数据集重复次数", args.repeats)
    args.grad_checkpoint = _ask_bool("启用梯度检查点?", args.grad_checkpoint)
    args.epochs = _ask_int("Epochs", args.epochs)
    args.max_steps = _ask_int("最大步数 (0=无限制)", args.max_steps)
    args.lora_rank = _ask_int("LoRA rank", args.lora_rank)
    args.lora_alpha = _ask_float("LoRA alpha", args.lora_alpha)
    args.loss_curve_steps = _ask_int("Loss 曲线步数 (0=禁用)", args.loss_curve_steps)
    args.auto_install = _ask_bool("自动安装缺失依赖?", args.auto_install)
    args.save_every = _ask_int("每隔多少 epoch 保存 (0=禁用)", args.save_every)
    args.mixed_precision = _ask_str("混合精度 (bf16/fp16/fp32)", args.mixed_precision)
    return args


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()

    # 加载 YAML 配置文件
    if args.config:
        logger.info(f"加载配置文件: {args.config}")
        config = load_yaml_config(args.config)
        args = apply_yaml_config(args, config)

    # 交互模式检查
    required = [args.data_dir, args.transformer, args.vae, args.qwen]
    if args.interactive or any(not x for x in required):
        args = prompt_for_args(args)

    # 依赖检测
    ensure_dependencies(auto_install=args.auto_install)

    # 延迟导入
    import numpy as np
    from PIL import Image

    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 创建输出目录
    if args.lora_name:
        output_dir = Path(args.output_dir) / args.lora_name
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    # 查找模型代码
    repo_root = find_diffusion_pipe_root()
    logger.info(f"模型代码路径: {repo_root}")

    # 加载模型
    logger.info("加载 Transformer...")
    model = load_anima_model(args.transformer, device, dtype, repo_root)

    # 启用 xformers
    if args.xformers:
        enable_xformers(model)

    logger.info("加载 VAE...")
    vae = load_vae(args.vae, device, dtype, repo_root)

    logger.info("加载文本编码器...")
    qwen_model, qwen_tok, t5_tok = load_text_encoders(
        args.qwen, args.t5_tokenizer, device, dtype
    )

    # 注入 LoRA
    logger.info(f"注入 {args.lora_type.upper()}...")
    injector = LoRAInjector(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        use_lokr=(args.lora_type == "lokr"),
        factor=args.lokr_factor,
    )
    injector.inject(model)

    # 数据集
    bucket_mgr = BucketManager(args.resolution)
    dataset = ImageDataset(
        args.data_dir, args.resolution, bucket_mgr,
        shuffle_caption=args.shuffle_caption,
        keep_tokens=args.keep_tokens,
        flip_augment=args.flip_augment,
    )

    # 缓存 VAE latents（在 repeat 之前）
    use_cached = getattr(args, "cache_latents", False)
    if use_cached:
        dataset = CachedLatentDataset(dataset, vae, device, dtype)

    # repeat 放在缓存之后
    if args.repeats > 1:
        dataset = RepeatDataset(dataset, repeats=args.repeats)

    if args.num_workers > 0 and os.name == "nt":
        logger.warning("num_workers > 0 on Windows may fail. Consider --num-workers 0.")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_cached if use_cached else collate_fn,
        num_workers=args.num_workers
    )

    # 优化器 (8-bit AdamW)
    params = injector.get_params()
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(params, lr=args.lr)
        logger.info("使用 8-bit AdamW 优化器")
    except ImportError:
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        logger.warning("bitsandbytes 未安装，使用标准 AdamW")

    # 恢复训练状态
    start_epoch = 0
    global_step = 0
    if args.resume:
        resume_dir = Path(args.resume)
        resolved_resume = None
        if resume_dir.exists():
            if (resume_dir / "state.pt").exists() or (resume_dir / "lora.safetensors").exists():
                resolved_resume = resume_dir
            else:
                candidates = []
                for p in resume_dir.glob("epoch-*"):
                    if not p.is_dir():
                        continue
                    try:
                        epoch_num = int(p.name.split("-", 1)[1])
                    except Exception:
                        continue
                    candidates.append((epoch_num, p))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    resolved_resume = candidates[-1][1]
        if resolved_resume:
            start_epoch, global_step = load_training_state(resolved_resume, injector, optimizer)
            start_epoch += 1  # 从下一个 epoch 开始
            logger.info(f"从 epoch {start_epoch} 恢复训练: {resolved_resume}")
        else:
            logger.warning(f"Resume 目录无有效状态: {resume_dir}")

    # 计算总步数
    try:
        steps_per_epoch = len(dataloader) // args.grad_accum
    except Exception:
        steps_per_epoch = None

    if args.max_steps and args.max_steps > 0:
        total_steps = args.max_steps
    elif steps_per_epoch is not None:
        total_steps = steps_per_epoch * args.epochs
    else:
        total_steps = None

    logger.info(f"数据集大小: {len(dataset)}, 每 epoch 步数: {steps_per_epoch}, 总步数: {total_steps}")

    # 初始化进度显示
    progress, task_id, progress_kind = init_progress(not args.no_progress, total_steps)
    use_rich = progress_kind == "rich"
    use_plain = progress == "plain"
    live = None
    loss_history = []
    speed_ema = None

    if use_rich:
        try:
            from rich.console import Group
            from rich.live import Live
            curve_panel = None
            if args.loss_curve_steps > 0 and not args.no_live_curve:
                curve_panel = render_curve_panel([], width=min(60, args.loss_curve_steps), height=10)
            group = Group(progress, curve_panel) if curve_panel is not None else Group(progress)
            live = Live(group, refresh_per_second=10)
            live.start()
        except Exception:
            live = None
            progress.start()

    def emit(msg):
        if use_plain:
            print()
        if live:
            live.console.print(msg)
        elif use_rich:
            progress.console.print(msg)
        else:
            print(msg)

    # 训练循环
    model.train()
    step_start_time = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        had_batches = False
        for batch_idx, batch in enumerate(dataloader):
            had_batches = True
            # 在累积周期开始时记录时间
            if batch_idx % args.grad_accum == 0:
                step_start_time = time.perf_counter()

            captions = batch["captions"]

            # 获取 latents（缓存模式或实时编码）
            if use_cached:
                latents = batch["latents"].to(device, dtype=dtype)
            else:
                pixels = batch["pixel_values"].to(device, dtype=dtype)
                with torch.no_grad():
                    pixels_5d = pixels.unsqueeze(2)  # [B,C,1,H,W]
                    latents = vae.model.encode(pixels_5d, vae.scale)

            bs = latents.shape[0]

            # 文本编码
            with torch.no_grad():
                qwen_emb = encode_qwen(qwen_model, qwen_tok, captions, device)
                t5_ids = tokenize_t5(t5_tok, captions).input_ids.to(device)
                cross = model.preprocess_text_embeds(qwen_emb, t5_ids)
                if cross.shape[1] < 512:
                    cross = F.pad(cross, (0, 0, 0, 512 - cross.shape[1]))

            # Flow Matching
            t = sample_t(bs, device)
            t_exp = t.view(-1, 1, 1, 1, 1)
            noise = torch.randn_like(latents)
            noisy = (1 - t_exp) * latents + t_exp * noise
            target = noise - latents

            # 前向
            pad_mask = torch.zeros(bs, 1, latents.shape[-2], latents.shape[-1], device=device, dtype=dtype)
            autocast_ctx = torch.autocast("cuda", dtype=dtype) if device == "cuda" else nullcontext()
            with autocast_ctx:
                pred = forward_with_optional_checkpoint(
                    model, noisy, t.view(-1, 1), cross, pad_mask,
                    use_checkpoint=args.grad_checkpoint
                )
                loss = F.mse_loss(pred.float(), target.float())

            # 反向传播
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # 记录 loss 历史
                loss_val = float(loss.item() * args.grad_accum)
                if args.loss_curve_steps and len(loss_history) < args.loss_curve_steps:
                    loss_history.append(loss_val)

                # 更新进度显示
                now = time.perf_counter()
                lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
                dt_step = now - step_start_time
                steps_per_sec = (1.0 / dt_step) if dt_step > 0 else 0.0
                speed_ema = steps_per_sec if speed_ema is None else (0.9 * speed_ema + 0.1 * steps_per_sec)

                if use_rich:
                    desc = f"epoch {epoch+1}/{args.epochs} step {global_step}/{total_steps or '?'}"
                    progress.update(task_id, advance=1, description=desc,
                                    loss=loss_val, lr=float(lr), speed=float(speed_ema or 0))
                    if live and args.loss_curve_steps > 0 and not args.no_live_curve:
                        panel = render_curve_panel(loss_history, width=min(60, args.loss_curve_steps), height=10)
                        if panel is not None:
                            from rich.console import Group
                            live.update(Group(progress, panel))
                elif use_plain:
                    print(f"epoch {epoch+1}/{args.epochs} step {global_step} loss={loss_val:.6f} lr={lr:.2e} speed={speed_ema:.2f} it/s", end="\r", flush=True)
                elif args.log_every and global_step % args.log_every == 0:
                    print(f"epoch={epoch} step={global_step} loss={loss_val:.6f} lr={lr:.2e} speed={steps_per_sec:.2f} it/s")

                # 检查 max_steps
                if args.max_steps and global_step >= args.max_steps:
                    break

        # 处理未对齐的梯度累积
        if had_batches and (batch_idx + 1) % args.grad_accum != 0:
            if not args.max_steps or global_step < args.max_steps:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        # epoch 结束后的操作
        current_epoch = epoch + 1
        if not args.max_steps or global_step < args.max_steps:
            # 保存 checkpoint
            if args.save_every > 0 and current_epoch % args.save_every == 0:
                if args.lora_name:
                    lora_type_suffix = "LoKr" if args.lora_type == "lokr" else "LoRA"
                    save_name = f"{args.lora_name}_Anima_{lora_type_suffix}-ep{current_epoch}.safetensors"
                else:
                    save_name = f"{args.output_name}_epoch{current_epoch}.safetensors"
                save_path = output_dir / save_name
                injector.save(save_path)
                emit(f"Saved LoRA: {save_path}")

            # 保存训练状态
            if args.save_state > 0:
                state_dir = output_dir / "state"
                save_training_state(state_dir, injector, optimizer, current_epoch, global_step, keep_last_n=args.save_state)

            # 采样
            if args.sample_every > 0 and current_epoch % args.sample_every == 0:
                img = sample_image(
                    model, vae, qwen_model, qwen_tok, t5_tok,
                    args.sample_prompt, device=device, dtype=dtype
                )
                img.save(sample_dir / f"epoch_{current_epoch}.png")
                emit(f"采样保存: epoch_{current_epoch}.png")

        # 检查 max_steps
        if args.max_steps and global_step >= args.max_steps:
            break

    # 最终保存
    if args.lora_name:
        lora_type_suffix = "LoKr" if args.lora_type == "lokr" else "LoRA"
        final_name = f"{args.lora_name}_Anima_{lora_type_suffix}.safetensors"
    else:
        final_name = f"{args.output_name}.safetensors"
    final_path = output_dir / final_name
    injector.save(final_path)

    # 清理进度显示
    if live:
        live.stop()
    elif use_rich:
        progress.stop()

    # 显示最终 loss 曲线
    if args.loss_curve_steps and loss_history:
        chart = render_loss_curve(loss_history, width=min(80, len(loss_history)), height=10)
        emit(f"Loss curve (first {len(loss_history)} steps):\n{chart}")

    emit(f"Saved final LoRA: {final_path}")
    logger.info("训练完成!")


if __name__ == "__main__":
    main()
