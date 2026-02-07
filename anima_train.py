#!/usr/bin/env python
"""
Anima (Cosmos-Predict2) LoRA trainer v1.02 - standalone script for Windows.

新增功能:
  - TOML 配置文件支持 (--config)
  - 多 repeat 数据支持 (Kohya 风格目录命名: {repeat}_{concept})
  - 直接导出 ComfyUI 兼容格式 (--comfyui-format)
  - Latent 缓存支持 (--cache-latents)
  - xformers 支持 (--xformers)

Requirements:
  - torch (CUDA)
  - transformers
  - safetensors
  - Pillow, numpy
  - xformers (可选)

This script dynamically loads model code from diffusion-pipe-main if present.
Set DIFFUSION_PIPE_ROOT if your diffusion-pipe-main path is non-standard.
"""

import argparse
import math
import json
import logging
import os
import random
import re
import fnmatch
import sys
import types
import hashlib
from pathlib import Path
from typing import Optional

import subprocess
from collections import deque

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# xformers 检测
XFORMERS_AVAILABLE = False
try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
except ImportError:
    pass


def find_diffusion_pipe_root():
    env_root = os.environ.get("DIFFUSION_PIPE_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path(__file__).resolve().parent / "anima_vendor")
    candidates.append(Path(__file__).resolve().parent / "_deprecated" / "thisisref" / "diffusion-pipe-main")
    for candidate in candidates:
        if (candidate / "models" / "anima_modeling.py").exists():
            return candidate
    # Fallback: search within repo
    search_root = Path(__file__).resolve().parent
    for path in search_root.rglob("anima_modeling.py"):
        if path.name == "anima_modeling.py" and path.parent.name == "models":
            return path.parent.parent
    raise SystemExit("Could not find diffusion-pipe-main. Set DIFFUSION_PIPE_ROOT.")


def load_module_from_path(module_name, file_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_dependencies(auto_install=False):
    required = {
        "numpy": "numpy",
        "PIL": "Pillow",
        "safetensors": "safetensors",
        "transformers": "transformers",
        "einops": "einops",
        "torchvision": "torchvision",
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


def ensure_wandb(auto_install=False):
    try:
        import wandb  # noqa: F401
        return
    except Exception:
        pass
    print("Missing dependency: wandb")
    if not auto_install:
        print(f"Install it with:\n  {sys.executable} -m pip install wandb")
        raise SystemExit(1)
    cmd = [sys.executable, "-m", "pip", "install", "wandb"]
    print("Installing wandb...")
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"wandb auto-install failed: {exc}")
        raise SystemExit(1)


def ensure_bitsandbytes(auto_install=False):
    try:
        import bitsandbytes  # noqa: F401
        return
    except Exception:
        if not auto_install:
            print(
                "Missing dependency: bitsandbytes\n"
                f"Install it with:\n  {sys.executable} -m pip install bitsandbytes"
            )
            raise SystemExit(1)
    cmd = [sys.executable, "-m", "pip", "install", "bitsandbytes"]
    print("Installing bitsandbytes...")
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"bitsandbytes auto-install failed: {exc}")
        raise SystemExit(1)
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        print("bitsandbytes install did not make it importable. Please install a compatible wheel manually.")
        raise SystemExit(1)


def ensure_prodigy(auto_install=False):
    # Common SD ecosystem name is `prodigyopt`.
    try:
        import prodigyopt  # noqa: F401
        return
    except Exception:
        if not auto_install:
            print(
                "Missing dependency: prodigyopt\n"
                f"Install it with:\n  {sys.executable} -m pip install prodigyopt"
            )
            raise SystemExit(1)
    cmd = [sys.executable, "-m", "pip", "install", "prodigyopt"]
    print("Installing prodigyopt...")
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"prodigyopt auto-install failed: {exc}")
        raise SystemExit(1)
    try:
        import prodigyopt  # noqa: F401
    except Exception:
        print("prodigyopt install did not make it importable. Please install manually.")
        raise SystemExit(1)


def ensure_lycoris(auto_install=False):
    try:
        import lycoris  # noqa: F401
        return
    except Exception:
        if not auto_install:
            print(
                "Missing dependency: lycoris-lora\n"
                f"Install it with:\n  {sys.executable} -m pip install lycoris-lora"
            )
            raise SystemExit(1)
    cmd = [sys.executable, "-m", "pip", "install", "lycoris-lora"]
    print("Installing lycoris-lora...")
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"lycoris-lora auto-install failed: {exc}")
        raise SystemExit(1)
    try:
        import lycoris  # noqa: F401
    except Exception:
        print("lycoris-lora install did not make it importable. Please install manually.")
        raise SystemExit(1)


def ensure_models_namespace(repo_root):
    repo_root = Path(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    models_path = repo_root / "models"
    if "models" not in sys.modules:
        models_pkg = types.ModuleType("models")
        models_pkg.__path__ = [str(models_path)]
        sys.modules["models"] = models_pkg
    else:
        pkg = sys.modules["models"]
        if not hasattr(pkg, "__path__"):
            pkg.__path__ = [str(models_path)]
        elif str(models_path) not in pkg.__path__:
            pkg.__path__.append(str(models_path))


def load_anima_classes(repo_root):
    ensure_models_namespace(repo_root)
    cosmos_modeling = load_module_from_path(
        "cosmos_predict2_modeling",
        repo_root / "models" / "cosmos_predict2_modeling.py",
    )
    sys.modules["models.cosmos_predict2_modeling"] = cosmos_modeling
    anima_modeling = load_module_from_path(
        "anima_modeling",
        repo_root / "models" / "anima_modeling.py",
    )
    return anima_modeling.Anima


def load_wan_vae(repo_root):
    wan_vae = load_module_from_path(
        "wan_vae2_1",
        repo_root / "models" / "wan" / "vae2_1.py",
    )
    return wan_vae.WanVAE_


def load_config_from_ckpt(ckpt_path):
    from safetensors import safe_open
    # Minimal clone of diffusion-pipe get_dit_config()
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        key = None
        for k in f.keys():
            if k.endswith("x_embedder.proj.1.weight"):
                key = k
                break
        if key is None:
            raise RuntimeError("Could not find x_embedder.proj.1.weight in checkpoint")
        w = f.get_tensor(key)
    concat_padding_mask = True
    in_channels = (w.shape[1] // 4) - int(concat_padding_mask)
    model_channels = w.shape[0]
    if model_channels == 2048:
        num_blocks = 28
        num_heads = 16
    elif model_channels == 5120:
        num_blocks = 36
        num_heads = 40
    else:
        raise RuntimeError(f"Unexpected model_channels={model_channels}")
    if in_channels == 16:
        rope_h_extrapolation_ratio = 4.0
        rope_w_extrapolation_ratio = 4.0
    elif in_channels == 17:
        rope_h_extrapolation_ratio = 3.0
        rope_w_extrapolation_ratio = 3.0
    else:
        rope_h_extrapolation_ratio = 1.0
        rope_w_extrapolation_ratio = 1.0

    return dict(
        max_img_h=240,
        max_img_w=240,
        max_frames=128,
        in_channels=in_channels,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=concat_padding_mask,
        model_channels=model_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        min_fps=1,
        max_fps=30,
        use_adaln_lora=True,
        adaln_lora_dim=256,
        rope_h_extrapolation_ratio=rope_h_extrapolation_ratio,
        rope_w_extrapolation_ratio=rope_w_extrapolation_ratio,
        rope_t_extrapolation_ratio=1.0,
        extra_per_block_abs_pos_emb=False,
        extra_h_extrapolation_ratio=1.0,
        extra_w_extrapolation_ratio=1.0,
        extra_t_extrapolation_ratio=1.0,
        rope_enable_fps_modulation=False,
    )


def load_state_dict(path):
    from safetensors import safe_open
    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            key = k
            if key.startswith("net."):
                key = key[len("net.") :]
            if key.startswith("diffusion_model."):
                key = key[len("diffusion_model.") :]
            sd[key] = f.get_tensor(k)
    return sd


def load_anima_model(ckpt_path, device, dtype, repo_root):
    Anima = load_anima_classes(repo_root)
    config = load_config_from_ckpt(ckpt_path)
    model = Anima(**config)
    sd = load_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"WARNING: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device=device, dtype=dtype)
    return model


def load_vae(vae_path, device, dtype, repo_root):
    from safetensors import safe_open
    WanVAE_ = load_wan_vae(repo_root)
    cfg = dict(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    model = WanVAE_(**cfg).eval().requires_grad_(False)
    sd = {}
    with safe_open(vae_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"WARNING: VAE missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device=device, dtype=dtype)

    class _VAEWrapper:
        pass

    wrapper = _VAEWrapper()
    wrapper.model = model
    mean = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ]
    std = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ]
    wrapper.mean = torch.tensor(mean, dtype=dtype, device=device)
    wrapper.std = torch.tensor(std, dtype=dtype, device=device)
    wrapper.scale = [wrapper.mean, 1.0 / wrapper.std]
    return wrapper


def load_qwen(qwen_path, device, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(qwen_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        config=config,
        dtype=dtype,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval().requires_grad_(False)
    model = model.to(device)
    return tokenizer, model


def load_t5_tokenizer(repo_root, t5_dir=None):
    from transformers import T5TokenizerFast

    if t5_dir is None:
        t5_dir = repo_root / "configs" / "t5_old"
    t5_dir = Path(t5_dir)
    return T5TokenizerFast(
        vocab_file=str(t5_dir / "spiece.model"),
        tokenizer_file=str(t5_dir / "tokenizer.json"),
    )


def tokenize_qwen(tokenizer, captions, max_length=512):
    return tokenizer(
        captions,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def tokenize_t5(tokenizer, captions, max_length=512):
    return tokenizer(
        captions,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def init_progress(show_progress, total_steps):
    if not show_progress:
        return None, None, None
    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TextColumn("lr={task.fields[lr]:.2e}"),
            TextColumn("speed={task.fields[speed]:.3f} it/s"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=10,
            # Default speed estimate window is 30s in Rich. For very slow iterations
            # (e.g. < 1 step / 30s), ETA remains unknown. Use a wider window.
            speed_estimate_period=600.0,
        )
        task = progress.add_task("train", total=total_steps, loss=0.0, lr=0.0, speed=0.0)
        return progress, task, "rich"
    except Exception:
        return "plain", None, None


def render_loss_curve(losses, width=60, height=10):
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
    try:
        from rich.panel import Panel
        from rich.text import Text
    except Exception:
        return None


def _maybe_init_wandb(args, logger):
    if not getattr(args, "wandb", False):
        return None
    mode = getattr(args, "wandb_mode", "online") or "online"
    if mode == "disabled":
        return None

    ensure_wandb(auto_install=args.auto_install)
    import wandb

    project = args.wandb_project or "anima-trainer"
    entity = args.wandb_entity or None
    # Default run name to output_name so W&B runs match LoRA outputs without extra config.
    name = args.wandb_name or args.output_name or None
    group = args.wandb_group or None
    tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]

    wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        tags=tags if tags else None,
        mode=mode,
        config={
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "output_name": args.output_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "optimizer": getattr(args, "optimizer", "adamw"),
            "weight_decay": getattr(args, "weight_decay", 0.0),
            "mixed_precision": args.mixed_precision,
            "resolution": args.resolution,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": args.lora_targets,
            "network_type": getattr(args, "network_type", "lora"),
            "lokr_factor": getattr(args, "lokr_factor", 8),
            "lokr_use_tucker": getattr(args, "lokr_use_tucker", False),
            "lokr_decompose_both": getattr(args, "lokr_decompose_both", False),
            "lokr_full_matrix": getattr(args, "lokr_full_matrix", False),
            "lokr_rank_dropout": getattr(args, "lokr_rank_dropout", 0.0),
            "lokr_module_dropout": getattr(args, "lokr_module_dropout", 0.0),
            "lokr_constraint": getattr(args, "lokr_constraint", 0.0),
            "lokr_normalize": getattr(args, "lokr_normalize", False),
            "shuffle_caption": args.shuffle_caption,
            "shuffle_caption_per_epoch": getattr(args, "shuffle_caption_per_epoch", False),
            "text_cache_size": getattr(args, "text_cache_size", 0),
            "cache_latents": args.cache_latents,
            "xformers": args.xformers,
            "grad_checkpoint": args.grad_checkpoint,
            "monitor_enabled": getattr(args, "monitor_enabled", True),
            "monitor_every": getattr(args, "monitor_every", 5),
            "monitor_memory": getattr(args, "monitor_memory", True),
            "monitor_wandb": getattr(args, "monitor_wandb", True),
            "monitor_alert_policy": getattr(args, "monitor_alert_policy", "warn"),
        },
    )
    logger.info("wandb enabled: project=%s mode=%s", project, mode)
    return wandb


class TrainingMonitor:
    def __init__(self, args, logger, wandb_obj, total_steps, start_time, device):
        self.enabled = bool(getattr(args, "monitor_enabled", True))
        self.every = max(1, int(getattr(args, "monitor_every", 5)))
        self.monitor_memory = bool(getattr(args, "monitor_memory", True))
        self.monitor_wandb = bool(getattr(args, "monitor_wandb", True))
        self.alert_policy = str(getattr(args, "monitor_alert_policy", "warn") or "warn").lower()
        self.logger = logger
        self.wandb = wandb_obj
        self.total_steps = int(total_steps)
        self.start_time = float(start_time)
        self.device = device
        self.loss_window = deque(maxlen=8)
        self.memory_warn_ratio = 0.95
        self.total_memory_bytes = None

        wandb_enabled = bool(getattr(args, "wandb", False))
        self.can_log_wandb = self.enabled and self.monitor_wandb and wandb_enabled and (wandb_obj is not None)

        if self.monitor_memory and torch.cuda.is_available():
            try:
                dev_index = device.index if isinstance(device, torch.device) and device.index is not None else torch.cuda.current_device()
                self.total_memory_bytes = float(torch.cuda.get_device_properties(dev_index).total_memory)
            except Exception:
                self.total_memory_bytes = None

    @staticmethod
    def _fmt_value(value):
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if math.isfinite(value):
                return f"{value:.6g}"
            return str(value)
        return str(value)

    def collect_step_metrics(
        self,
        *,
        epoch,
        global_step,
        step_in_epoch,
        loss,
        lr,
        speed_it_s,
        samples_per_s,
        grad_accum,
        optimizer_name,
        prodigy_d,
        grad_nonzero_ratio,
    ):
        metrics = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "total_steps": int(self.total_steps),
            "step_in_epoch": int(step_in_epoch),
            "loss": float(loss),
            "lr": float(lr),
            "speed_it_s": float(speed_it_s),
            "samples_per_s": float(samples_per_s),
            "grad_accum": int(grad_accum),
            "optimizer": str(optimizer_name),
        }
        if prodigy_d is not None:
            metrics["prodigy_d"] = float(prodigy_d)
        if grad_nonzero_ratio is not None:
            metrics["grad_nonzero_ratio"] = float(grad_nonzero_ratio)

        if self.monitor_memory and torch.cuda.is_available():
            try:
                alloc = torch.cuda.memory_allocated(device=self.device) / (1024.0 * 1024.0)
                reserved = torch.cuda.memory_reserved(device=self.device) / (1024.0 * 1024.0)
                max_alloc = torch.cuda.max_memory_allocated(device=self.device) / (1024.0 * 1024.0)
                max_reserved = torch.cuda.max_memory_reserved(device=self.device) / (1024.0 * 1024.0)
                metrics["mem_allocated_mb"] = float(alloc)
                metrics["mem_reserved_mb"] = float(reserved)
                metrics["mem_max_allocated_mb"] = float(max_alloc)
                metrics["mem_max_reserved_mb"] = float(max_reserved)
            except Exception:
                pass

        return metrics

    def emit_console(self, metrics):
        ordered_keys = [
            "epoch",
            "global_step",
            "total_steps",
            "step_in_epoch",
            "loss",
            "lr",
            "speed_it_s",
            "samples_per_s",
            "grad_accum",
            "optimizer",
            "prodigy_d",
            "grad_nonzero_ratio",
            "mem_allocated_mb",
            "mem_reserved_mb",
            "mem_max_allocated_mb",
            "mem_max_reserved_mb",
            "alert_nonfinite_loss",
            "alert_memory_near_limit",
            "alert_loss_spike",
            "alert_any",
        ]
        parts = []
        for key in ordered_keys:
            if key in metrics:
                parts.append(f"{key}={self._fmt_value(metrics[key])}")
        self.logger.info("[monitor] %s", " ".join(parts))

    def emit_wandb(self, metrics):
        if not self.can_log_wandb:
            return
        payload = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                payload[f"monitor/{key}"] = int(value)
            elif isinstance(value, (int, float, str)):
                payload[f"monitor/{key}"] = value
        if not payload:
            return
        try:
            self.wandb.log(payload, step=int(metrics.get("global_step", 0)))
        except Exception as exc:
            self.logger.warning("wandb.log failed: %s", exc)

    def check_alerts(self, metrics):
        alert_nonfinite_loss = False
        alert_memory_near_limit = False
        alert_loss_spike = False

        loss = metrics.get("loss")
        try:
            loss_val = float(loss)
        except Exception:
            loss_val = float("nan")
        if not math.isfinite(loss_val):
            alert_nonfinite_loss = True
            self.logger.warning("[monitor] non-finite loss at step %s: %s", metrics.get("global_step"), loss)
        else:
            self.loss_window.append(loss_val)
            if len(self.loss_window) >= 5:
                baseline = list(self.loss_window)[:-1]
                baseline_sorted = sorted(baseline)
                baseline_median = baseline_sorted[len(baseline_sorted) // 2]
                if baseline_median > 0 and loss_val > (baseline_median * 3.0):
                    alert_loss_spike = True
                    self.logger.warning(
                        "[monitor] loss spike at step %s: loss=%.6g baseline_median=%.6g",
                        metrics.get("global_step"),
                        loss_val,
                        baseline_median,
                    )

        if self.monitor_memory and self.total_memory_bytes:
            alloc_mb = float(metrics.get("mem_allocated_mb", 0.0) or 0.0)
            reserved_mb = float(metrics.get("mem_reserved_mb", 0.0) or 0.0)
            total_mb = self.total_memory_bytes / (1024.0 * 1024.0)
            if total_mb > 0:
                alloc_ratio = alloc_mb / total_mb
                reserved_ratio = reserved_mb / total_mb
                if alloc_ratio >= self.memory_warn_ratio or reserved_ratio >= self.memory_warn_ratio:
                    alert_memory_near_limit = True
                    self.logger.warning(
                        "[monitor] memory near limit at step %s: alloc=%.1fMB reserved=%.1fMB total=%.1fMB",
                        metrics.get("global_step"),
                        alloc_mb,
                        reserved_mb,
                        total_mb,
                    )

        metrics["alert_nonfinite_loss"] = alert_nonfinite_loss
        metrics["alert_memory_near_limit"] = alert_memory_near_limit
        metrics["alert_loss_spike"] = alert_loss_spike
        metrics["alert_any"] = bool(alert_nonfinite_loss or alert_memory_near_limit or alert_loss_spike)

    def log_step_if_needed(
        self,
        *,
        epoch,
        global_step,
        step_in_epoch,
        loss,
        lr,
        speed_it_s,
        samples_per_s,
        grad_accum,
        optimizer_name,
        prodigy_d,
        grad_nonzero_ratio,
    ):
        if not self.enabled:
            return
        if int(global_step) % self.every != 0:
            return

        metrics = self.collect_step_metrics(
            epoch=epoch,
            global_step=global_step,
            step_in_epoch=step_in_epoch,
            loss=loss,
            lr=lr,
            speed_it_s=speed_it_s,
            samples_per_s=samples_per_s,
            grad_accum=grad_accum,
            optimizer_name=optimizer_name,
            prodigy_d=prodigy_d,
            grad_nonzero_ratio=grad_nonzero_ratio,
        )
        self.check_alerts(metrics)
        self.emit_console(metrics)
        self.emit_wandb(metrics)


def xformers_attention(q, k, v):
    """使用 xformers 的 memory efficient attention"""
    if not XFORMERS_AVAILABLE:
        raise RuntimeError("xformers not available")
    # q, k, v: [B, S, H, D] -> [B, H, S, D] for xformers
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    out = xformers.ops.memory_efficient_attention(q, k, v)
    # [B, H, S, D] -> [B, S, H*D]
    B, H, S, D = out.shape
    return out.transpose(1, 2).reshape(B, S, H * D)


def compute_qwen_embeddings(qwen_model, input_ids, attention_mask):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    input_ids = input_ids.to(qwen_model.device, dtype=torch.long)
    attention_mask = attention_mask.to(qwen_model.device, dtype=torch.long)
    with torch.no_grad():
        outputs = qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states[-1]
    lengths = attention_mask.sum(dim=1).cpu()
    for batch_id in range(hidden_states.shape[0]):
        length = lengths[batch_id]
        if length == 1:
            length = 0
        hidden_states[batch_id][length:] = 0
    return hidden_states


def sample_t(bs, device, method="logit_normal", sigmoid_scale=1.0, shift=3.0):
    if method == "logit_normal":
        dist = torch.distributions.normal.Normal(0, 1)
    elif method == "uniform":
        dist = torch.distributions.uniform.Uniform(0, 1)
    else:
        raise ValueError(f"Unknown method {method}")
    t = dist.sample((bs,)).to(device)
    if method == "logit_normal":
        t = torch.sigmoid(t * sigmoid_scale)
    if shift is not None:
        t = (t * shift) / (1 + (shift - 1) * t)
    return t


def forward_with_optional_checkpoint(model, latents, timesteps, cross, padding_mask, use_checkpoint=False):
    if not use_checkpoint:
        return model(latents, timesteps, cross, padding_mask=padding_mask)
    from torch.utils.checkpoint import checkpoint

    x_B_T_H_W_D, rope_emb, extra_pos_emb = model.prepare_embedded_sequence(
        latents,
        fps=None,
        padding_mask=padding_mask,
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


def _stable_caption_hash_int(text: str) -> int:
    if not text:
        return 0
    d = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(d[:8], byteorder="little", signed=False)


class BucketManager:
    def __init__(
        self,
        base_resolution=1024,
        min_resolution=512,
        max_resolution=2048,
        resolution_step=64,
        max_aspect_ratio=2.0,
    ):
        self.base_resolution = base_resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.resolution_step = resolution_step
        # Allow disabling AR filtering via 0/negative values.
        self.max_aspect_ratio = None if (max_aspect_ratio is None or max_aspect_ratio <= 0) else float(max_aspect_ratio)
        self.buckets = self._generate_buckets()

    def _generate_buckets(self):
        buckets = []
        base_area = self.base_resolution ** 2
        for w in range(self.min_resolution, self.max_resolution + 1, self.resolution_step):
            for h in range(self.min_resolution, self.max_resolution + 1, self.resolution_step):
                area = w * h
                if abs(area - base_area) / base_area > 0.1:
                    continue
                aspect = max(w / h, h / w)
                if self.max_aspect_ratio is not None and aspect > self.max_aspect_ratio:
                    continue
                buckets.append((w, h))
        return buckets

    def get_bucket(self, width, height):
        aspect = width / height
        best_bucket = (self.base_resolution, self.base_resolution)
        best_diff = float("inf")
        for bw, bh in self.buckets:
            diff = abs(aspect - (bw / bh))
            if diff < best_diff:
                best_diff = diff
                best_bucket = (bw, bh)
        return best_bucket


class LatentCache:
    """Latent 缓存管理器"""
    def __init__(self, cache_dir: str, vae, device, dtype, *, cache_namespace: str = "legacy", cache_meta: dict | None = None):
        self.cache_root = Path(cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        ns = re.sub(r"[^0-9A-Za-z._-]", "_", str(cache_namespace or "legacy"))
        self.cache_namespace = ns or "legacy"
        self.cache_dir = self.cache_root / self.cache_namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vae = vae
        self.device = device
        self.dtype = dtype
        self.logger = logging.getLogger(__name__)
        if cache_meta is not None:
            try:
                meta_path = self.cache_dir / "_cache_meta.json"
                meta_path.write_text(
                    json.dumps(cache_meta, ensure_ascii=False, sort_keys=True, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                self.logger.warning("Failed to write cache metadata: %s", exc)

    def _get_cache_path(self, image_path: str) -> Path:
        """获取缓存文件路径"""
        import hashlib
        hash_key = hashlib.md5(image_path.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.pt"

    def has_cache(self, image_path: str) -> bool:
        """检查是否有缓存"""
        return self._get_cache_path(image_path).exists()

    def load_cache(self, image_path: str) -> torch.Tensor:
        """加载缓存的 latent"""
        cache_path = self._get_cache_path(image_path)
        return torch.load(cache_path, map_location="cpu")

    def save_cache(self, image_path: str, latent: torch.Tensor):
        """保存 latent 到缓存"""
        cache_path = self._get_cache_path(image_path)
        torch.save(latent.cpu(), cache_path)


class CaptionDataset(Dataset):
    """支持 Kohya 风格目录命名的数据集类。"""
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(
        self,
        data_dir,
        resolution=1024,
        caption_extension=".txt",
        enable_bucket=True,
        min_bucket_reso=512,
        max_bucket_reso=2048,
        bucket_reso_steps=64,
        max_bucket_aspect_ratio=2.0,
        shuffle_caption=False,
        keep_tokens=0,
        flip_augment=False,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.caption_extension = caption_extension
        self.shuffle_caption = shuffle_caption
        self.shuffle_caption_per_epoch = False
        self._epoch = 0
        self._caption_seed = 0
        self.keep_tokens = keep_tokens
        self.flip_augment = flip_augment
        self.transform = transform

        self.logger = logging.getLogger(__name__)

        if enable_bucket:
            self.bucket_manager = BucketManager(
                base_resolution=resolution,
                min_resolution=min_bucket_reso,
                max_resolution=max_bucket_reso,
                resolution_step=bucket_reso_steps,
                max_aspect_ratio=max_bucket_aspect_ratio,
            )
        else:
            self.bucket_manager = None

        self.samples = self._scan_dataset()
        self.logger.info("Found %d samples in %s", len(self.samples), data_dir)

    def set_epoch(self, epoch: int, *, seed: int = 0):
        """
        Set epoch index for deterministic per-epoch caption shuffling.

        When enabled, captions for the same image stay stable within an epoch and change across epochs.
        """
        self._epoch = int(epoch)
        self._caption_seed = int(seed)

    def _parse_kohya_dir_name(self, dir_name: str) -> tuple:
        """
        解析 Kohya 风格目录名: {repeat}_{concept}
        返回 (repeat, concept) 或 (1, None) 如果不匹配
        """
        match = re.match(r'^(\d+)_(.+)$', dir_name)
        if match:
            return int(match.group(1)), match.group(2)
        return 1, None

    def _scan_dataset(self):
        from PIL import Image
        samples = []
        if not self.data_dir.exists():
            self.logger.warning("Dataset folder not found: %s", self.data_dir)
            return samples

        paths = sorted(self.data_dir.rglob("*"), key=lambda p: p.as_posix())
        for img_path in paths:
            try:
                if not img_path.is_file() or img_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue

                # 解析 Kohya 风格目录名获取 repeat
                parent_dir = img_path.parent
                if parent_dir != self.data_dir:
                    repeat, concept = self._parse_kohya_dir_name(parent_dir.name)
                else:
                    repeat, concept = 1, None

                sample = self._process_image_file(img_path, repeat, concept)
                if sample:
                    samples.append(sample)
            except Exception as exc:
                self.logger.warning("Error processing sample %s: %s", img_path, exc)
                continue
        return samples

    def _process_image_file(self, img_path: Path, repeat: int = 1, concept: str = None) -> dict:
        """处理单个图像文件，返回样本字典"""
        from PIL import Image

        caption_path = img_path.with_suffix(self.caption_extension)
        if not caption_path.exists():
            caption_path = img_path.with_suffix(".txt")

        caption = ""
        if caption_path.exists():
            try:
                caption = caption_path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                self.logger.warning("Failed to read caption for %s: %s", img_path, exc)

        try:
            with Image.open(img_path) as img:
                original_size = img.size
        except Exception as exc:
            self.logger.warning("Failed to open %s: %s", img_path, exc)
            return None

        if self.bucket_manager:
            target_size = self.bucket_manager.get_bucket(*original_size)
        else:
            target_size = (self.resolution, self.resolution)

        crop_coords = self._calculate_crop(original_size, target_size)
        return {
            "image_path": str(img_path),
            "caption": caption,
            "original_size": original_size,
            "target_size": target_size,
            "crop_coords": crop_coords,
            "repeat": repeat,
            "concept": concept,
        }

    @staticmethod
    def _calculate_crop(original_size, target_size):
        ow, oh = original_size
        tw, th = target_size
        scale = max(tw / ow, th / oh)
        scaled_w = int(ow * scale)
        scaled_h = int(oh * scale)
        left = (scaled_w - tw) // 2
        top = (scaled_h - th) // 2
        return (left, top, left + tw, top + th)

    def _process_caption(self, caption):
        if not caption:
            return ""

        lines = [line.strip() for line in caption.splitlines() if line.strip()]
        if lines:
            tags_text = lines[0]
            nl_text = " ".join(lines[1:]).strip()
        else:
            tags_text = caption.strip()
            nl_text = ""

        if "," in tags_text:
            tags = [t.strip() for t in tags_text.split(",") if t.strip()]
        else:
            tags = [t.strip() for t in tags_text.split() if t.strip()]

        # Kohya-style behavior: when enabled, shuffle tag tokens per sample fetch (effectively per step).
        rng = random

        if self.keep_tokens > 0:
            kept = tags[: self.keep_tokens]
            rest = tags[self.keep_tokens :]
            if self.shuffle_caption:
                rng.shuffle(rest)
            tags = kept + rest
        elif self.shuffle_caption:
            rng.shuffle(tags)

        tags_out = ", ".join(tags)
        if tags_out and nl_text:
            return f"{tags_out}, {nl_text}"
        if nl_text:
            return nl_text
        return tags_out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        tw, th = sample["target_size"]
        scale = max(tw / image.width, th / image.height)
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        image = image.crop((left, top, left + tw, top + th))

        if self.flip_augment and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        caption = self._process_caption(sample["caption"])

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image = image * 2.0 - 1.0

        return {
            "image": image,
            "caption": caption,
            "original_size": sample["original_size"],
            "target_size": sample["target_size"],
            "crop_coords": sample["crop_coords"],
            "repeat": sample.get("repeat", 1),
        }


def create_dataloader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, *, seed=None):
    def _seed_worker(_worker_id: int):
        worker_seed = int(torch.initial_seed() % (2**32))
        random.seed(worker_seed)
        try:
            import numpy as np

            np.random.seed(worker_seed)
        except Exception:
            pass
        try:
            torch.manual_seed(worker_seed)
        except Exception:
            pass

    def _get_sample_for_index(ds, idx):
        # KohyaRepeatDataset
        if hasattr(ds, "index_map"):
            base_idx = ds.index_map[idx]
            return ds.dataset.samples[base_idx]
        # RepeatDataset
        if hasattr(ds, "repeats") and hasattr(ds, "dataset"):
            base_len = len(ds.dataset)
            return ds.dataset.samples[idx % base_len]
        # LatentCacheDataset or base dataset
        if hasattr(ds, "samples"):
            return ds.samples[idx]
        return None

    def _build_bucketed_batches(ds, bs, do_shuffle):
        rng = random.Random(int(seed)) if seed is not None else random
        buckets = {}
        for idx in range(len(ds)):
            sample = _get_sample_for_index(ds, idx)
            if not sample:
                continue
            target_size = sample.get("target_size")
            buckets.setdefault(target_size, []).append(idx)
        batches = []
        for indices in buckets.values():
            if do_shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), bs):
                batch = indices[i:i + bs]
                if batch:
                    batches.append(batch)
        if do_shuffle:
            rng.shuffle(batches)
        return batches

    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        captions = [item["caption"] for item in batch]
        original_sizes = [item["original_size"] for item in batch]
        target_sizes = [item["target_size"] for item in batch]
        crop_coords = [item["crop_coords"] for item in batch]
        result = {
            "images": images,
            "captions": captions,
            "original_sizes": original_sizes,
            "target_sizes": target_sizes,
            "crop_coords": crop_coords,
        }
        # 支持 latent 缓存
        if "latent" in batch[0]:
            result["latent"] = [item["latent"] for item in batch]
            result["use_cached_latent"] = [item.get("use_cached_latent", False) for item in batch]
        return result

    if batch_size > 1 and hasattr(dataset, "samples"):
        bucketed_batches = _build_bucketed_batches(dataset, batch_size, shuffle)
        if bucketed_batches:
            class _ListBatchSampler:
                def __init__(self, batches):
                    self.batches = batches

                def __iter__(self):
                    return iter(self.batches)

                def __len__(self):
                    return len(self.batches)

            return DataLoader(
                dataset,
                batch_sampler=_ListBatchSampler(bucketed_batches),
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
                worker_init_fn=_seed_worker if int(num_workers or 0) > 0 else None,
            )

    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        generator=g,
        worker_init_fn=_seed_worker if int(num_workers or 0) > 0 else None,
    )


class RepeatDataset(Dataset):
    """简单的重复数据集包装器"""
    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    @property
    def samples(self):
        """代理访问底层数据集的 samples"""
        return self.dataset.samples

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class KohyaRepeatDataset(Dataset):
    """
    支持 per-sample repeat 的数据集包装器 (Kohya 风格)。
    根据每个样本的 repeat 字段构建展开后的索引映射。
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.index_map = self._build_index_map()

    @property
    def samples(self):
        """代理访问底层数据集的 samples"""
        return self.dataset.samples

    def _build_index_map(self):
        """构建展开后的索引映射"""
        index_map = []
        for i in range(len(self.dataset)):
            sample = self.dataset.samples[i]
            repeat = sample.get("repeat", 1)
            for _ in range(repeat):
                index_map.append(i)
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        real_idx = self.index_map[idx]
        return self.dataset[real_idx]


class LatentCacheDataset(Dataset):
    """支持 latent 缓存的数据集包装器"""
    def __init__(self, dataset, cache: LatentCache):
        self.dataset = dataset
        self.cache = cache

    @property
    def samples(self):
        """代理访问底层数据集的 samples"""
        return self.dataset.samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = self.samples[idx]["image_path"]

        if self.cache.has_cache(image_path):
            item["latent"] = self.cache.load_cache(image_path)
            item["use_cached_latent"] = True
        else:
            item["use_cached_latent"] = False
        return item


class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.lora_down = torch.nn.Linear(in_features, self.rank, bias=False)
        self.lora_up = torch.nn.Linear(self.rank, out_features, bias=False)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling


def _factorization(dimension: int, factor: int) -> tuple[int, int]:
    """
    Align factorization behavior with LyCORIS/ai-toolkit.
    Returns (m, n) with m <= n.
    """
    dimension = max(1, int(dimension))
    factor = int(factor)
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n

    if factor < 0:
        factor = dimension

    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n

    if m > n:
        n, m = m, n
    return m, n


class LoKrLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank=4,
        alpha=1.0,
        factor=8,
        full_matrix=False,
        decompose_both=False,
        rank_dropout=0.0,
        module_dropout=0.0,
        constraint=0.0,
        normalize=False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.factor = max(1, int(factor))
        self.full_matrix = bool(full_matrix)
        self.decompose_both = bool(decompose_both)
        self.rank_dropout = float(rank_dropout)
        self.module_dropout = float(module_dropout)
        self.constraint = float(constraint)
        self.normalize = bool(normalize)

        self.out_l, self.out_k = _factorization(self.out_features, self.factor)
        self.in_m, self.in_n = _factorization(self.in_features, self.factor)

        # LyCORIS-compatible branch selection for linear LoKr.
        w1_threshold = max(self.out_l, self.in_m) / 2.0
        w2_threshold = max(self.out_k, self.in_n) / 2.0
        self.w1_direct = self.full_matrix or (not (self.decompose_both and self.rank < w1_threshold))
        self.w2_direct = self.full_matrix or (self.rank >= w2_threshold)

        # LyCORIS alpha behavior: alpha==0 means "use rank"; full/full uses scale=1.
        effective_alpha = self.alpha if self.alpha != 0 else float(self.rank)
        if self.w1_direct and self.w2_direct:
            effective_alpha = float(self.rank)
        self.scaling = effective_alpha / float(self.rank)

        if self.w1_direct:
            self.lokr_w1 = torch.nn.Parameter(torch.empty(self.out_l, self.in_m))
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=5 ** 0.5)
        else:
            self.lokr_w1_a = torch.nn.Parameter(torch.empty(self.out_l, self.rank))
            self.lokr_w1_b = torch.nn.Parameter(torch.empty(self.rank, self.in_m))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=5 ** 0.5)
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=5 ** 0.5)

        if self.w2_direct:
            # 与 lycoris/kohya 对齐：大分支置零，保持初始增量为 0。
            self.lokr_w2 = torch.nn.Parameter(torch.zeros(self.out_k, self.in_n))
        else:
            # w2_a 随机、w2_b 置零：既保证初始输出不变，也避免双零死锁。
            self.lokr_w2_a = torch.nn.Parameter(torch.empty(self.out_k, self.rank))
            self.lokr_w2_b = torch.nn.Parameter(torch.zeros(self.rank, self.in_n))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=5 ** 0.5)

    def _w1(self):
        if self.w1_direct:
            return self.lokr_w1
        return self.lokr_w1_a @ self.lokr_w1_b

    def _w2(self):
        if self.w2_direct:
            return self.lokr_w2
        return self.lokr_w2_a @ self.lokr_w2_b

    def _diff_weight(self):
        w1 = self._w1()
        w2 = self._w2()
        diff = torch.kron(w1, w2).reshape(self.out_features, self.in_features) * self.scaling
        if self.normalize:
            n = diff.norm()
            if n > 0:
                diff = diff / n
        if self.constraint > 0:
            n = diff.norm()
            if n > self.constraint:
                diff = diff * (self.constraint / (n + 1e-12))
        return diff

    def forward(self, x):
        if self.training and self.module_dropout > 0 and random.random() < self.module_dropout:
            return torch.zeros((*x.shape[:-1], self.out_features), device=x.device, dtype=x.dtype)
        diff = self._diff_weight().to(device=x.device, dtype=x.dtype)
        if self.training and self.rank_dropout > 0:
            mask = (torch.rand(diff.shape[0], device=diff.device) > self.rank_dropout).to(diff.dtype)
            diff = diff * mask[:, None]
        return F.linear(x, diff)


class LoRALinear(torch.nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.original = original_layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.lora.to(device=original_layer.weight.device, dtype=original_layer.weight.dtype)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora(x)


class LoKrLinear(torch.nn.Module):
    def __init__(self, original_layer, *, rank, alpha, dropout, lokr_factor, lokr_full_matrix, lokr_decompose_both, lokr_rank_dropout, lokr_module_dropout, lokr_constraint, lokr_normalize):
        super().__init__()
        self.original = original_layer
        self.input_dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.lokr = LoKrLayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            factor=lokr_factor,
            full_matrix=lokr_full_matrix,
            decompose_both=lokr_decompose_both,
            rank_dropout=lokr_rank_dropout,
            module_dropout=lokr_module_dropout,
            constraint=lokr_constraint,
            normalize=lokr_normalize,
        )
        self.lokr.to(device=original_layer.weight.device, dtype=original_layer.weight.dtype)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lokr(self.input_dropout(x))


# ComfyUI 格式转换所需的已知层名模式
KNOWN_PATTERNS = [
    "llm_adapter",
    "blocks", "self_attn", "cross_attn", "q_proj", "k_proj", "v_proj",
    "output_proj", "o_proj", "mlp", "layer1", "layer2",
    "adaln_modulation_self_attn", "adaln_modulation_cross_attn", "adaln_modulation_mlp",
    "layer_norm_self_attn", "layer_norm_cross_attn", "layer_norm_mlp",
]


_ADAPTER_SUFFIXES = [
    ".lora_down.weight",
    ".lora_up.weight",
    ".lokr_w1",
    ".lokr_w2",
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_t2",
]

ANIMA_TARGET_PRESETS = ("v101", "anima_block", "anima_block_llm")

# Legacy default targets from v1.01/v1.02.
DEFAULT_TARGETS_V101 = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.k_proj",
    "cross_attn.v_proj",
    "cross_attn.o_proj",
    "cross_attn.output_proj",
    "mlp.layer1",
    "mlp.layer2",
    "mlp.0",
    "mlp.2",
]


def _collect_linear_targets_by_owner_class(model, owner_class_names):
    targets = set()
    for owner_name, owner in model.named_modules():
        if owner.__class__.__name__ not in owner_class_names:
            continue
        for sub_name, submodule in owner.named_modules():
            if not sub_name:
                continue
            if not isinstance(submodule, torch.nn.Linear):
                continue
            full_name = f"{owner_name}.{sub_name}" if owner_name else sub_name
            targets.add(full_name)
    return sorted(targets)


def resolve_target_modules(args, model, logger):
    if args.lora_targets:
        target_modules = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        if not target_modules:
            raise ValueError("--lora-targets is set but no valid non-empty target was provided")
        logger.info("Using custom --lora-targets (%d entries)", len(target_modules))
        return target_modules

    preset = str(getattr(args, "anima_target_preset", "v101") or "v101").lower()
    if preset == "v101":
        logger.info("Using target preset: v101 (%d entries)", len(DEFAULT_TARGETS_V101))
        return list(DEFAULT_TARGETS_V101)

    if preset == "anima_block":
        classes = {"Block"}
    elif preset == "anima_block_llm":
        # Align with diffusion-pipe style adapter coverage for cosmos/anima.
        classes = {"Block", "TransformerBlock"}
    else:
        raise ValueError(
            f"Unknown --anima-target-preset={preset}. Expected one of: {', '.join(ANIMA_TARGET_PRESETS)}"
        )

    target_modules = _collect_linear_targets_by_owner_class(model, classes)
    if not target_modules:
        raise RuntimeError(
            f"Preset {preset} resolved 0 target modules (classes={sorted(classes)}). "
            "Check the loaded model architecture."
        )
    logger.info(
        "Using target preset: %s (%d linear modules from classes=%s)",
        preset,
        len(target_modules),
        ",".join(sorted(classes)),
    )
    return target_modules


def _split_adapter_key(old_key: str):
    for suffix in _ADAPTER_SUFFIXES:
        if old_key.endswith(suffix):
            return old_key[:-len(suffix)], suffix
    return old_key, None


def convert_key_to_comfyui(old_key: str) -> str:
    """转换键名为 ComfyUI 兼容格式"""
    if old_key.startswith("diffusion_model.") and (".lora_" in old_key or ".lokr_" in old_key):
        return old_key.replace("diffusion_model.llm.adapter.", "diffusion_model.llm_adapter.")

    base, suffix = _split_adapter_key(old_key)
    if suffix is None:
        return old_key

    if base.startswith("transformer_"):
        base = base[len("transformer_"):]

    result_parts = []
    remaining = base

    while remaining:
        matched = False
        num_match = re.match(r'^(\d+)_?', remaining)
        if num_match:
            result_parts.append(num_match.group(1))
            remaining = remaining[len(num_match.group(0)):]
            matched = True
            continue

        for pattern in sorted(KNOWN_PATTERNS, key=len, reverse=True):
            if remaining.startswith(pattern):
                result_parts.append(pattern)
                remaining = remaining[len(pattern):]
                if remaining.startswith("_"):
                    remaining = remaining[1:]
                matched = True
                break

        if not matched:
            idx = remaining.find("_")
            if idx == -1:
                result_parts.append(remaining)
                remaining = ""
            else:
                result_parts.append(remaining[:idx])
                remaining = remaining[idx + 1:]

    new_base = ".".join(result_parts)
    new_key = f"diffusion_model.{new_base}{suffix}"
    return new_key.replace("diffusion_model.llm.adapter.", "diffusion_model.llm_adapter.")


class _BaseInjector:
    network_type = "lora"

    def __init__(self, rank=32, alpha=16.0, dropout=0.0, target_modules=None):
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.target_modules = [t for t in (target_modules or []) if t]
        self.injected_layers = {}
        self.training_metadata = {}

    def _should_inject(self, name: str, module: torch.nn.Module) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        layer_name = name.split(".")[-1]
        for raw_target in self.target_modules:
            target = str(raw_target or "").strip()
            if not target:
                continue
            if target.startswith("transformer."):
                target = target[len("transformer.") :]

            # regex: re:<pattern>
            if target.startswith("re:"):
                if re.search(target[3:], name):
                    return True
                continue

            # wildcard path match
            if "*" in target or "?" in target:
                if fnmatch.fnmatch(name, target):
                    return True
                continue

            # Dotted target is treated as path suffix with boundary.
            # e.g. "mlp.2" matches "...mlp.2" but not "...adaln_modulation_mlp.2".
            if "." in target:
                if name == target or name.endswith("." + target):
                    return True
                continue

            # Single token target matches only the leaf module name.
            if layer_name == target:
                return True
        return False

    def _wrap_layer(self, module: torch.nn.Linear):
        raise NotImplementedError

    def inject_model(self, model, prefix=""):
        injected = {}
        for name, module in model.named_modules():
            if not self._should_inject(name, module):
                continue
            wrapped = self._wrap_layer(module)
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, wrapped)
            full_name = f"{prefix}.{name}" if prefix else name
            injected[full_name] = wrapped
            self.injected_layers[full_name] = wrapped
        return injected

    def get_trainable_params(self):
        params = []
        for layer in self.injected_layers.values():
            if hasattr(layer, "lora"):
                params.extend(layer.lora.parameters())
            elif hasattr(layer, "lokr"):
                params.extend(layer.lokr.parameters())
        return params

    def _layer_state_pairs(self, _name: str, _layer) -> list[tuple[str, torch.Tensor]]:
        raise NotImplementedError

    def state_dict(self):
        sd = {}
        for name, layer in self.injected_layers.items():
            base = name.replace(".", "_")
            for suffix, tensor in self._layer_state_pairs(name, layer):
                sd[f"{base}{suffix}"] = tensor.data
        return sd

    def _expected_tensors(self):
        expected = {}
        for name, layer in self.injected_layers.items():
            base = name.replace(".", "_")
            for suffix, tensor in self._layer_state_pairs(name, layer):
                expected[f"{base}{suffix}"] = tensor
        return expected

    def load_state_dict(self, sd: dict, *, strict: bool = True):
        if not isinstance(sd, dict):
            raise TypeError(f"{self.network_type} state_dict must be a dict")

        expected = self._expected_tensors()

        missing = [k for k in expected.keys() if k not in sd]
        unexpected = [k for k in sd.keys() if k not in expected]

        if strict and missing:
            raise RuntimeError(f"{self.network_type} resume checkpoint is missing {len(missing)} keys (e.g. {missing[0]})")
        if strict and unexpected:
            raise RuntimeError(f"{self.network_type} resume checkpoint has {len(unexpected)} unexpected keys (e.g. {unexpected[0]})")
        if unexpected and not strict:
            print(f"WARNING: {self.network_type} resume checkpoint has {len(unexpected)} unexpected keys (ignored)")

        loaded = 0
        for k, param in expected.items():
            if k not in sd:
                continue
            t = sd[k]
            if not torch.is_tensor(t):
                raise TypeError(f"{self.network_type} tensor for {k} is not a Tensor")
            param.data.copy_(t.to(device=param.device, dtype=param.dtype))
            loaded += 1

        if loaded == 0:
            raise RuntimeError(f"No {self.network_type} weights were loaded from resume checkpoint (0 tensors)")
        return {"loaded": loaded, "missing": len(missing), "unexpected": len(unexpected)}

    def _save_metadata(self, comfyui: bool = False):
        base = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_output_name": "anima_lora",
            "network_type": self.network_type,
        }
        if self.network_type == "lokr":
            base["ss_network_module"] = "lycoris.kohya"
        else:
            base["ss_network_module"] = "networks.lora"
        base["format"] = "comfyui_compatible" if comfyui else f"anima_internal_{self.network_type}"
        if self.training_metadata:
            base.update(self.training_metadata)
        return {k: str(v) for k, v in base.items()}

    def set_training_metadata(self, args, target_modules):
        def _normalize(v):
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            if isinstance(v, (list, tuple)):
                return [_normalize(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _normalize(val) for k, val in v.items()}
            return str(v)

        args_dict = {}
        for k, v in vars(args).items():
            args_dict[str(k)] = _normalize(v)

        self.training_metadata = {
            "ss_training_args": json.dumps(args_dict, ensure_ascii=False, sort_keys=True),
            "ss_target_modules": json.dumps(list(target_modules or []), ensure_ascii=False),
            "ss_training_network_type": str(getattr(args, "network_type", self.network_type)),
            "ss_trainer_impl": "anima_train.py_v1.02",
        }

    def save(self, path):
        from safetensors.torch import save_file
        save_file(self.state_dict(), path, metadata=self._save_metadata(comfyui=False))

    def save_comfyui(self, path):
        """保存为 ComfyUI 兼容格式"""
        from safetensors.torch import save_file

        sd = self.state_dict()
        new_weights = {}
        converted_layers = set()

        for old_key, tensor in sd.items():
            new_key = convert_key_to_comfyui(old_key)
            if new_key in new_weights:
                raise RuntimeError(f"ComfyUI key collision: {new_key}")
            new_weights[new_key] = tensor
            if ".lora_" in new_key or ".lokr_" in new_key:
                layer_name = new_key.rsplit(".", 1)[0]
                converted_layers.add(layer_name)

        # 为每个层添加 alpha 值
        for layer_name in converted_layers:
            alpha_key = f"{layer_name}.alpha"
            if alpha_key in new_weights:
                continue
            new_weights[alpha_key] = torch.tensor(float(self.alpha), dtype=torch.float32)

        save_file(new_weights, path, metadata=self._save_metadata(comfyui=True))


class LoRAInjector(_BaseInjector):
    network_type = "lora"

    def _wrap_layer(self, module: torch.nn.Linear):
        return LoRALinear(module, rank=self.rank, alpha=self.alpha, dropout=self.dropout)

    def _layer_state_pairs(self, _name: str, layer):
        return [
            (".lora_down.weight", layer.lora.lora_down.weight),
            (".lora_up.weight", layer.lora.lora_up.weight),
        ]


class LoKrInjector(_BaseInjector):
    network_type = "lokr"

    def __init__(
        self,
        rank=32,
        alpha=16.0,
        dropout=0.0,
        target_modules=None,
        lokr_factor=8,
        lokr_full_matrix=False,
        lokr_use_tucker=False,
        lokr_decompose_both=False,
        lokr_rank_dropout=0.0,
        lokr_module_dropout=0.0,
        lokr_constraint=0.0,
        lokr_normalize=False,
    ):
        super().__init__(rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules)
        self.lokr_factor = int(lokr_factor)
        self.lokr_full_matrix = bool(lokr_full_matrix)
        self.lokr_use_tucker = bool(lokr_use_tucker)
        self.lokr_decompose_both = bool(lokr_decompose_both)
        self.lokr_rank_dropout = float(lokr_rank_dropout)
        self.lokr_module_dropout = float(lokr_module_dropout)
        self.lokr_constraint = float(lokr_constraint)
        self.lokr_normalize = bool(lokr_normalize)

    def _wrap_layer(self, module: torch.nn.Linear):
        return LoKrLinear(
            module,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            lokr_factor=self.lokr_factor,
            lokr_full_matrix=self.lokr_full_matrix,
            lokr_decompose_both=self.lokr_decompose_both,
            lokr_rank_dropout=self.lokr_rank_dropout,
            lokr_module_dropout=self.lokr_module_dropout,
            lokr_constraint=self.lokr_constraint,
            lokr_normalize=self.lokr_normalize,
        )

    def _layer_state_pairs(self, _name: str, layer):
        pairs = []
        lokr = layer.lokr
        if hasattr(lokr, "lokr_w1"):
            pairs.append((".lokr_w1", lokr.lokr_w1))
        if hasattr(lokr, "lokr_w2"):
            pairs.append((".lokr_w2", lokr.lokr_w2))
        if hasattr(lokr, "lokr_w1_a"):
            pairs.append((".lokr_w1_a", lokr.lokr_w1_a))
        if hasattr(lokr, "lokr_w1_b"):
            pairs.append((".lokr_w1_b", lokr.lokr_w1_b))
        if hasattr(lokr, "lokr_w2_a"):
            pairs.append((".lokr_w2_a", lokr.lokr_w2_a))
        if hasattr(lokr, "lokr_w2_b"):
            pairs.append((".lokr_w2_b", lokr.lokr_w2_b))
        return pairs

    def _save_metadata(self, comfyui: bool = False):
        meta = super()._save_metadata(comfyui=comfyui)
        meta["ss_network_args"] = str(
            {
                "algo": "lokr",
                "factor": self.lokr_factor,
                "full_matrix": self.lokr_full_matrix,
                "use_tucker": self.lokr_use_tucker,
                "decompose_both": self.lokr_decompose_both,
                "rank_dropout": self.lokr_rank_dropout,
                "module_dropout": self.lokr_module_dropout,
                "constraint": self.lokr_constraint,
                "normalize": self.lokr_normalize,
            }
        )
        return meta


def build_network_injector(args, target_modules):
    network_type = str(getattr(args, "network_type", "lora") or "lora").lower()
    if network_type == "lokr":
        ensure_lycoris(auto_install=getattr(args, "auto_install", False))
        return LoKrInjector(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
            lokr_factor=getattr(args, "lokr_factor", 8),
            lokr_full_matrix=getattr(args, "lokr_full_matrix", False),
            lokr_use_tucker=getattr(args, "lokr_use_tucker", False),
            lokr_decompose_both=getattr(args, "lokr_decompose_both", False),
            lokr_rank_dropout=getattr(args, "lokr_rank_dropout", 0.0),
            lokr_module_dropout=getattr(args, "lokr_module_dropout", 0.0),
            lokr_constraint=getattr(args, "lokr_constraint", 0.0),
            lokr_normalize=getattr(args, "lokr_normalize", False),
        )
    return LoRAInjector(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules,
    )


def _optimizer_state_to_device(optimizer, device):
    # torch.optim does not guarantee moving state tensors to the param device on load.
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device=device)


def build_optimizer(args, trainable_params, logger):
    opt_name = (getattr(args, "optimizer", "adamw") or "adamw").lower()
    betas = (float(getattr(args, "beta1", 0.9)), float(getattr(args, "beta2", 0.999)))
    eps = float(getattr(args, "eps", 1e-8))
    wd = float(getattr(args, "weight_decay", 0.01))

    if opt_name == "adamw":
        return torch.optim.AdamW(trainable_params, lr=args.lr, betas=betas, eps=eps, weight_decay=wd)

    if opt_name == "adamw8bit":
        ensure_bitsandbytes(auto_install=args.auto_install)
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise RuntimeError(f"bitsandbytes import failed: {exc}")
        return bnb.optim.AdamW8bit(trainable_params, lr=args.lr, betas=betas, eps=eps, weight_decay=wd)

    if opt_name == "prodigy":
        ensure_prodigy(auto_install=args.auto_install)
        try:
            from prodigyopt import Prodigy
        except Exception as exc:
            raise RuntimeError(f"prodigyopt import failed: {exc}")
        beta3_raw = float(getattr(args, "prodigy_beta3", 0.999))
        beta3 = None if beta3_raw <= 0 else beta3_raw
        d0 = float(getattr(args, "prodigy_d0", 1e-6))
        d_coef = float(getattr(args, "prodigy_d_coef", 1.0))
        growth_rate = float(getattr(args, "prodigy_growth_rate", float("inf")))
        slice_p = int(getattr(args, "prodigy_slice_p", 1))
        decouple = bool(getattr(args, "prodigy_decouple", True))
        use_bias_correction = bool(getattr(args, "prodigy_use_bias_correction", True))
        safeguard_warmup = bool(getattr(args, "prodigy_safeguard_warmup", True))
        # Handle minor API differences across prodigyopt versions by filtering kwargs by signature.
        import inspect

        sig = inspect.signature(Prodigy.__init__)
        params = sig.parameters

        kwargs = {}
        if "lr" in params:
            kwargs["lr"] = args.lr
        if "eps" in params:
            kwargs["eps"] = eps
        if "weight_decay" in params:
            kwargs["weight_decay"] = wd
        if "d0" in params:
            kwargs["d0"] = d0
        if "d_coef" in params:
            kwargs["d_coef"] = d_coef
        if "decouple" in params:
            kwargs["decouple"] = decouple
        if "use_bias_correction" in params:
            kwargs["use_bias_correction"] = use_bias_correction
        if "safeguard_warmup" in params:
            kwargs["safeguard_warmup"] = safeguard_warmup
        if "growth_rate" in params:
            kwargs["growth_rate"] = growth_rate
        if "slice_p" in params:
            kwargs["slice_p"] = slice_p

        if "beta3" in params:
            kwargs["betas"] = (betas[0], betas[1])
            kwargs["beta3"] = beta3
        else:
            beta3_fallback = beta3 if beta3 is not None else math.sqrt(betas[1])
            kwargs["betas"] = (betas[0], betas[1], beta3_fallback)

        return Prodigy(trainable_params, **kwargs)

    raise ValueError(f"Unknown optimizer: {opt_name}")


def _get_rng_state():
    rng = {
        "python_random": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            rng["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            rng["torch_cuda"] = None
    else:
        rng["torch_cuda"] = None
    return rng


def _set_rng_state(rng):
    if not rng:
        return
    try:
        if "python_random" in rng and rng["python_random"] is not None:
            random.setstate(rng["python_random"])
    except Exception:
        pass
    try:
        if "torch_cpu" in rng and rng["torch_cpu"] is not None:
            torch.set_rng_state(rng["torch_cpu"])
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            if "torch_cuda" in rng and rng["torch_cuda"]:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception:
            pass


def save_training_state(
    path,
    *,
    lora_injector,
    optimizer,
    scaler,
    next_epoch,
    global_step,
    losses,
    args,
):
    state = {
        "format_version": 1,
        "next_epoch": int(next_epoch),
        "global_step": int(global_step),
        "lora_state": lora_injector.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "losses_tail": losses[-200:],
        "rng_state": _get_rng_state(),
        "args_snapshot": {
            "network_type": str(getattr(args, "network_type", "lora") or "lora"),
            "lora_rank": int(args.lora_rank),
            "lora_alpha": float(args.lora_alpha),
            "lora_targets": str(args.lora_targets or ""),
            "lokr_factor": int(getattr(args, "lokr_factor", 8)),
            "lokr_use_tucker": bool(getattr(args, "lokr_use_tucker", False)),
            "lokr_decompose_both": bool(getattr(args, "lokr_decompose_both", False)),
            "lokr_full_matrix": bool(getattr(args, "lokr_full_matrix", False)),
            "lokr_rank_dropout": float(getattr(args, "lokr_rank_dropout", 0.0) or 0.0),
            "lokr_module_dropout": float(getattr(args, "lokr_module_dropout", 0.0) or 0.0),
            "lokr_constraint": float(getattr(args, "lokr_constraint", 0.0) or 0.0),
            "lokr_normalize": bool(getattr(args, "lokr_normalize", False)),
            "grad_accum": int(args.grad_accum),
            "mixed_precision": str(args.mixed_precision),
            "optimizer": str(getattr(args, "optimizer", "adamw") or "adamw"),
            "lr": float(getattr(args, "lr", 0.0) or 0.0),
            "weight_decay": float(getattr(args, "weight_decay", 0.0) or 0.0),
            "beta1": float(getattr(args, "beta1", 0.9) or 0.9),
            "beta2": float(getattr(args, "beta2", 0.999) or 0.999),
            "eps": float(getattr(args, "eps", 1e-8) or 1e-8),
        },
    }

    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_training_state(path):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise RuntimeError("Unsupported or invalid training state file")
    return state

def parse_args():
    parser = argparse.ArgumentParser()
    # v1.02 新增参数
    parser.add_argument("--config", default="", help="TOML 配置文件路径")
    parser.add_argument("--comfyui-format", action="store_true", help="直接导出 ComfyUI 兼容格式")
    parser.add_argument("--shuffle-caption", action="store_true", help="打乱 caption 标签顺序")
    parser.add_argument(
        "--shuffle-caption-per-epoch",
        action="store_true",
        help="Shuffle captions deterministically per epoch (stable within epoch, changes across epochs)",
    )
    parser.add_argument("--keep-tokens", type=int, default=0, help="保留前 N 个 token 不打乱")
    parser.add_argument("--flip-augment", action="store_true", help="启用水平翻转增强")
    parser.add_argument("--cache-latents", action="store_true", help="启用 latent 缓存")
    parser.add_argument("--cache-dir", default="", help="Latent 缓存目录 (默认: data_dir/.latent_cache)")
    parser.add_argument("--xformers", action="store_true", help="启用 xformers memory efficient attention")

    # 原有参数
    parser.add_argument("--data-dir", default="", help="Image folder with .txt captions")
    parser.add_argument("--transformer", default="", help="anima-preview.safetensors")
    parser.add_argument("--vae", default="", help="qwen_image_vae.safetensors")
    parser.add_argument("--qwen", default="", help="Qwen HF directory")
    parser.add_argument("--t5-tokenizer-dir", default="", help="Path to t5_old tokenizer dir")
    parser.add_argument("--output-dir", default="./output", help="Output dir")
    parser.add_argument("--output-name", default="anima_lora", help="LoRA name")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Clip grad norm for LoRA params (0=disable)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adamw8bit", "prodigy"],
        default="adamw",
        help="Optimizer type",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    # Prodigy extras (optional)
    parser.add_argument("--prodigy-beta3", type=float, default=0.999)
    parser.add_argument("--prodigy-d0", type=float, default=1e-6)
    parser.add_argument("--prodigy-d-coef", type=float, default=1.0)
    parser.add_argument(
        "--prodigy-growth-rate",
        type=float,
        default=float("inf"),
        help="Prodigy max multiplicative growth rate for d (default: inf)",
    )
    parser.add_argument(
        "--prodigy-slice-p",
        type=int,
        default=1,
        help="Prodigy memory reduction factor (official option; 1 means exact statistics)",
    )
    parser.add_argument(
        "--prodigy-decouple",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prodigy: decoupled weight decay",
    )
    parser.add_argument(
        "--prodigy-use-bias-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prodigy: bias correction",
    )
    parser.add_argument(
        "--prodigy-safeguard-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prodigy: safeguard warmup",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each sample N times")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--min-reso", type=int, default=512)
    parser.add_argument("--max-reso", type=int, default=2048)
    parser.add_argument("--reso-step", type=int, default=64)
    parser.add_argument("--max-ar", type=float, default=2.0)
    parser.add_argument("--mixed-precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--text-cache-size", type=int, default=256, help="In-epoch LRU cache size for text tokens/embeds (0=disable)")
    parser.add_argument("--resume", default="", help="Path to a .state.pt training checkpoint to resume from")
    parser.add_argument(
        "--save-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save paired .state.pt training checkpoints (enables resume)",
    )
    parser.add_argument(
        "--save-state-every",
        type=int,
        default=0,
        help=(
            "Save a state-only .state.pt every N epochs. "
            "When 0, save a paired .state.pt whenever saving a .safetensors checkpoint."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", default="")
    parser.add_argument(
        "--anima-target-preset",
        choices=list(ANIMA_TARGET_PRESETS),
        default="v101",
        help="Default Anima target module preset when --lora-targets is empty",
    )
    parser.add_argument("--network-type", choices=["lora", "lokr"], default="lora")
    parser.add_argument("--lokr-factor", type=int, default=8)
    parser.add_argument("--lokr-use-tucker", action="store_true")
    parser.add_argument("--lokr-decompose-both", action="store_true")
    parser.add_argument("--lokr-full-matrix", action="store_true")
    parser.add_argument("--lokr-rank-dropout", type=float, default=0.0)
    parser.add_argument("--lokr-module-dropout", type=float, default=0.0)
    parser.add_argument("--lokr-constraint", type=float, default=0.0)
    parser.add_argument("--lokr-normalize", action="store_true")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--loss-curve-steps", type=int, default=100)
    parser.add_argument("--no-live-curve", action="store_true")
    parser.add_argument("--auto-install", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    # Monitoring (detailed logs)
    parser.add_argument(
        "--monitor-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable unified training monitor logs",
    )
    parser.add_argument("--monitor-every", type=int, default=5, help="Emit monitor logs every N optimizer steps")
    parser.add_argument(
        "--monitor-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include CUDA memory metrics in monitor logs",
    )
    parser.add_argument(
        "--monitor-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send monitor metrics to W&B when wandb is enabled",
    )
    parser.add_argument(
        "--monitor-alert-policy",
        default="warn",
        help="Alert policy for monitor (currently warn-only; reserved for future)",
    )
    # W&B (optional)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="", help="wandb project name")
    parser.add_argument("--wandb-entity", default="", help="wandb entity/team (optional)")
    parser.add_argument("--wandb-name", default="", help="wandb run name (optional)")
    parser.add_argument("--wandb-group", default="", help="wandb group (optional)")
    parser.add_argument("--wandb-tags", default="", help="wandb tags comma-separated (optional)")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"], help="wandb mode")
    parser.add_argument("--wandb-log-every", type=int, default=1, help="wandb log frequency (optimizer steps)")
    return parser.parse_args()


def validate_args(args, logger):
    try:
        args.grad_accum = int(args.grad_accum)
    except Exception:
        logger.error("--grad-accum must be an integer")
        raise SystemExit(2)

    if args.grad_accum < 1:
        logger.error("--grad-accum must be >= 1")
        raise SystemExit(2)

    if int(args.batch_size) < 1:
        logger.error("--batch-size must be >= 1")
        raise SystemExit(2)

    if int(args.epochs) < 1:
        logger.error("--epochs must be >= 1")
        raise SystemExit(2)

    if int(args.seq_len) < 1:
        logger.error("--seq-len must be >= 1")
        raise SystemExit(2)

    mp = str(getattr(args, "mixed_precision", "") or "")
    if mp not in ("fp32", "fp16", "bf16"):
        logger.error("--mixed-precision must be one of: fp32, fp16, bf16 (got: %s)", mp)
        raise SystemExit(2)

    try:
        args.grad_clip = float(getattr(args, "grad_clip", 0.0) or 0.0)
    except Exception:
        logger.error("--grad-clip must be a number")
        raise SystemExit(2)

    if args.grad_clip < 0:
        logger.error("--grad-clip must be >= 0")
        raise SystemExit(2)

    try:
        args.monitor_every = int(getattr(args, "monitor_every", 5))
    except Exception:
        logger.error("--monitor-every must be an integer")
        raise SystemExit(2)
    if args.monitor_every < 1:
        logger.error("--monitor-every must be >= 1")
        raise SystemExit(2)

    args.monitor_alert_policy = str(getattr(args, "monitor_alert_policy", "warn") or "warn").lower()
    if args.monitor_alert_policy not in {"warn", "raise"}:
        logger.error("--monitor-alert-policy must be one of: warn, raise")
        raise SystemExit(2)
    if args.monitor_alert_policy == "raise":
        logger.warning("--monitor-alert-policy=raise is reserved; using warn-only behavior")

    if int(getattr(args, "lokr_factor", 8)) < 1:
        logger.error("--lokr-factor must be >= 1")
        raise SystemExit(2)
    for name in ("lokr_rank_dropout", "lokr_module_dropout"):
        value = float(getattr(args, name, 0.0) or 0.0)
        if value < 0.0 or value >= 1.0:
            logger.error("--%s must be in [0, 1)", name.replace("_", "-"))
            raise SystemExit(2)

    opt_name = str(getattr(args, "optimizer", "adamw") or "adamw").lower()
    if opt_name == "prodigy":
        try:
            slice_p = int(getattr(args, "prodigy_slice_p", 1))
        except Exception:
            logger.error("--prodigy-slice-p must be an integer")
            raise SystemExit(2)
        if slice_p < 1:
            logger.error("--prodigy-slice-p must be >= 1")
            raise SystemExit(2)
        try:
            growth_rate = float(getattr(args, "prodigy_growth_rate", float("inf")))
        except Exception:
            logger.error("--prodigy-growth-rate must be a number")
            raise SystemExit(2)
        if not math.isinf(growth_rate) and growth_rate <= 0:
            logger.error("--prodigy-growth-rate must be > 0 or inf")
            raise SystemExit(2)

        lr = float(getattr(args, "lr", 1.0))
        if lr <= 0.1:
            logger.warning(
                "Prodigy with lr=%.3g is likely too low; official recommendation is around 1.0 (tune d_coef instead).",
                lr,
            )


def _maybe_clip_grads(args, *, scaler, optimizer, trainable_params):
    max_norm = float(getattr(args, "grad_clip", 0.0) or 0.0)
    if max_norm <= 0:
        return
    if scaler is not None:
        try:
            scaler.unscale_(optimizer)
        except Exception:
            pass
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_norm)


def _check_adapter_gradients(trainable_params, logger, *, step_hint: int, remaining_checks: int) -> int:
    if remaining_checks <= 0:
        return 0
    total = 0
    with_grad = 0
    nonzero = 0
    max_abs = 0.0
    for p in trainable_params:
        total += 1
        g = getattr(p, "grad", None)
        if g is None:
            continue
        with_grad += 1
        try:
            cur = float(g.detach().abs().max().item())
        except Exception:
            cur = 0.0
        if cur > 0.0:
            nonzero += 1
        if cur > max_abs:
            max_abs = cur
    if nonzero > 0:
        logger.info(
            "Adapter gradient check passed at update %d (%d/%d tensors have non-zero grad, max_abs=%.3e)",
            step_hint,
            nonzero,
            total,
            max_abs,
        )
        # Keep printing until every trainable tensor has non-zero gradient.
        if nonzero >= total and total > 0:
            return 0
        return remaining_checks
    remaining = remaining_checks - 1
    logger.warning(
        "Adapter gradients are all zero at update %d (with_grad=%d/%d, checks_left=%d)",
        step_hint,
        with_grad,
        total,
        remaining,
    )
    if remaining <= 0:
        raise RuntimeError(
            "No non-zero adapter gradients detected in early training updates. "
            "This usually indicates target mismatch, detached forward path, or dead adapter init."
        )
    return remaining


def _compute_grad_nonzero_ratio(trainable_params) -> Optional[float]:
    total = 0
    nonzero = 0
    for p in trainable_params:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        total += 1
        try:
            cur = float(g.detach().abs().max().item())
        except Exception:
            cur = 0.0
        if cur > 0.0:
            nonzero += 1
    if total <= 0:
        return None
    return float(nonzero) / float(total)


def _get_current_lr(optimizer, fallback_lr: float) -> float:
    try:
        if optimizer is not None and optimizer.param_groups:
            return float(optimizer.param_groups[0].get("lr", fallback_lr))
    except Exception:
        pass
    return float(fallback_lr)


def _get_prodigy_d(optimizer) -> Optional[float]:
    try:
        if optimizer is None or not optimizer.param_groups:
            return None
        group = optimizer.param_groups[0]
        for key in ("d", "d_hat", "d_value", "d_t", "d_prev", "d0"):
            if key in group:
                return float(group[key])
    except Exception:
        return None
    return None


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
    defaults = _guess_default_paths()
    args.data_dir = args.data_dir or _ask_str("Dataset folder (images + .txt)", "")
    args.transformer = args.transformer or _ask_str("Anima transformer", defaults["transformer"])
    args.vae = args.vae or _ask_str("Qwen Image VAE", defaults["vae"])
    args.qwen = args.qwen or _ask_str("Qwen HF directory", defaults["qwen"])
    args.output_dir = _ask_str("Output dir", args.output_dir)
    args.output_name = _ask_str("Output name", args.output_name)
    args.resolution = _ask_int("Resolution", args.resolution)
    args.batch_size = _ask_int("Batch size", args.batch_size)
    args.grad_accum = _ask_int("Gradient accumulation", args.grad_accum)
    args.lr = _ask_float("Learning rate", args.lr)
    args.repeats = _ask_int("Dataset repeats", args.repeats)
    args.grad_checkpoint = _ask_bool("Enable gradient checkpointing?", args.grad_checkpoint)
    args.seq_len = _ask_int("Max token length", args.seq_len)
    args.epochs = _ask_int("Epochs", args.epochs)
    args.max_steps = _ask_int("Max steps (0 = no limit)", args.max_steps)
    args.num_workers = _ask_int("DataLoader workers", args.num_workers)
    args.lora_rank = _ask_int("LoRA rank", args.lora_rank)
    args.lora_alpha = _ask_float("LoRA alpha", args.lora_alpha)
    args.network_type = _ask_str("Network type (lora/lokr)", args.network_type)
    if str(args.network_type).lower() == "lokr":
        args.lokr_factor = _ask_int("LoKr factor", args.lokr_factor)
        args.lokr_rank_dropout = _ask_float("LoKr rank dropout", args.lokr_rank_dropout)
        args.lokr_module_dropout = _ask_float("LoKr module dropout", args.lokr_module_dropout)
        args.lokr_constraint = _ask_float("LoKr constraint", args.lokr_constraint)
        args.lokr_full_matrix = _ask_bool("LoKr full matrix?", args.lokr_full_matrix)
        args.lokr_decompose_both = _ask_bool("LoKr decompose both branches?", args.lokr_decompose_both)
        args.lokr_use_tucker = _ask_bool("LoKr use tucker?", args.lokr_use_tucker)
        args.lokr_normalize = _ask_bool("LoKr normalize diff?", args.lokr_normalize)
    # Loss curve display is disabled; keep args for compatibility but don't prompt.
    args.auto_install = _ask_bool("Auto install missing dependencies?", args.auto_install)
    args.save_every_epoch = _ask_bool("Save after each epoch?", args.save_every_epoch)
    args.mixed_precision = _ask_str("Mixed precision (bf16/fp16/fp32)", args.mixed_precision)
    return args


def detect_kohya_structure(data_dir: str) -> bool:
    """检测数据目录是否使用 Kohya 风格命名 ({repeat}_{concept})"""
    data_path = Path(data_dir)
    if not data_path.exists():
        return False

    for subdir in data_path.iterdir():
        if subdir.is_dir():
            match = re.match(r'^(\d+)_(.+)$', subdir.name)
            if match:
                return True
    return False


def build_latent_cache_namespace(args) -> tuple[str, dict]:
    # Any change in these fields invalidates old latent cache automatically.
    payload = {
        "cache_schema": "anima_latent_v2",
        "resolution": int(getattr(args, "resolution", 1024)),
        "min_reso": int(getattr(args, "min_reso", 512)),
        "max_reso": int(getattr(args, "max_reso", 2048)),
        "reso_step": int(getattr(args, "reso_step", 64)),
        "max_ar": float(getattr(args, "max_ar", 2.0)),
        "transformer": str(getattr(args, "transformer", "")),
        "vae": str(getattr(args, "vae", "")),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    namespace = f"cfg_{digest}"
    return namespace, payload


def cache_all_latents(dataset, vae, cache: LatentCache, device, dtype, logger):
    """预先缓存所有图像的 latent"""
    import numpy as np
    from PIL import Image
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    logger.info("开始缓存 latents...")
    cached_count = 0
    skipped_count = 0
    failed_count = 0
    total = len(dataset.samples)
    pbar = None
    if tqdm is not None and total > 0:
        pbar = tqdm(total=total, desc="Caching latents", unit="img", dynamic_ncols=True)

    try:
        for i, sample in enumerate(dataset.samples):
            image_path = sample["image_path"]

            if cache.has_cache(image_path):
                skipped_count += 1
                continue

            try:
                # 加载并处理图像
                image = Image.open(image_path).convert("RGB")
                tw, th = sample["target_size"]
                scale = max(tw / image.width, th / image.height)
                new_w = int(image.width * scale)
                new_h = int(image.height * scale)
                image = image.resize((new_w, new_h), Image.LANCZOS)
                left = (new_w - tw) // 2
                top = (new_h - th) // 2
                image = image.crop((left, top, left + tw, top + th))

                # 转换为 tensor
                img_tensor = torch.from_numpy(np.array(image))
                img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor * 2.0 - 1.0
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)
                img_tensor = img_tensor.to(device, dtype=dtype)

                # 编码
                with torch.no_grad():
                    latent = vae.model.encode(img_tensor, vae.scale)

                cache.save_cache(image_path, latent.squeeze(0))
                cached_count += 1

            except Exception as e:
                failed_count += 1
                logger.warning("缓存失败 %s: %s", image_path, e)

            if pbar is not None:
                pbar.update(1)
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    pbar.set_postfix(cached=cached_count, skipped=skipped_count, failed=failed_count)
            elif (i + 1) % 10 == 0:
                logger.info(
                    "  缓存进度: %d/%d (new=%d skip=%d fail=%d)",
                    i + 1,
                    total,
                    cached_count,
                    skipped_count,
                    failed_count,
                )
    finally:
        if pbar is not None:
            pbar.set_postfix(cached=cached_count, skipped=skipped_count, failed=failed_count)
            pbar.close()

    complete = (cached_count + skipped_count) == total
    logger.info(
        "缓存完成: 新缓存 %d, 跳过 %d, 失败 %d (总计 %d, 完整=%s)",
        cached_count,
        skipped_count,
        failed_count,
        total,
        "yes" if complete else "no",
    )
    return {
        "cached": cached_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "total": total,
        "complete": complete,
    }


def main():
    args = parse_args()

    # v1.02: 加载 TOML 配置
    if args.config:
        from config_loader import load_toml_config, apply_config_to_args
        print(f"Loading config from: {args.config}")
        config = load_toml_config(args.config)
        apply_config_to_args(args, config)

    if args.interactive:
        args = prompt_for_args(args)

    ensure_dependencies(auto_install=args.auto_install)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    wandb = None
    wandb_run = None

    # xformers 检查
    if args.xformers:
        if XFORMERS_AVAILABLE:
            logger.info("xformers 已启用")
        else:
            logger.warning("xformers 不可用，将使用 PyTorch 原生 attention")
            args.xformers = False

    if not args.data_dir:
        logger.error("--data-dir is required")
        return
    if not args.transformer:
        logger.error("--transformer is required")
        return

    validate_args(args, logger)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(args.mixed_precision, torch.bfloat16)

    logger.info("Device: %s, dtype: %s", device, dtype)

    repo_root = Path(__file__).resolve().parent
    logger.info("Loading Anima model from %s", args.transformer)
    model = load_anima_model(args.transformer, device, dtype, repo_root)
    model.train()
    # Match v1.01: freeze base weights; only LoRA params should be trainable.
    model.requires_grad_(False)

    vae = None
    if args.vae:
        logger.info("Loading VAE from %s", args.vae)
        vae = load_vae(args.vae, device, dtype, repo_root)

    qwen_tokenizer, qwen_model = None, None
    if args.qwen:
        logger.info("Loading Qwen from %s", args.qwen)
        qwen_tokenizer, qwen_model = load_qwen(args.qwen, device, dtype)

    # T5 tokenizer: v1.01 uses diffusion-pipe root's configs/t5_old by default.
    # trainerv1.02 directory may not contain configs/, so we resolve T5 relative to the pipe root.
    t5_tokenizer = None
    t5_dir = args.t5_tokenizer_dir if args.t5_tokenizer_dir else None
    try:
        pipe_root = find_diffusion_pipe_root()
        t5_tokenizer = load_t5_tokenizer(pipe_root, t5_dir)
    except Exception as exc:
        logger.warning("Could not load T5 tokenizer: %s", exc)
        t5_tokenizer = None
    if t5_tokenizer is None:
        logger.info("T5 tokenizer disabled/unavailable; training will use Qwen embeddings only (llm_adapter bypass).")
    else:
        logger.info("T5 tokenizer enabled.")

    # LoRA 注入
    target_modules = resolve_target_modules(args, model, logger)

    lora_injector = build_network_injector(args, target_modules)
    lora_injector.set_training_metadata(args, target_modules)
    lora_injector.inject_model(model, prefix="transformer")
    trainable_params = lora_injector.get_trainable_params()
    # Summarize injected layer coverage to catch mis-targeting early.
    try:
        from collections import Counter

        mod_counts = Counter()
        for full_name in lora_injector.injected_layers.keys():
            # full_name: transformer.<path.to.linear>
            mod_counts[full_name.split(".")[-1]] += 1
        top = ", ".join([f"{k}={v}" for k, v in mod_counts.most_common(10)])
        logger.info("Injected %d LoRA layers (%s)", len(lora_injector.injected_layers), top)
    except Exception:
        logger.info("Injected %d LoRA layers", len(lora_injector.injected_layers))
    logger.info(
        "Injected %s with rank=%d, alpha=%.1f, %d trainable params",
        str(getattr(args, "network_type", "lora")).upper(),
        args.lora_rank,
        args.lora_alpha,
        sum(p.numel() for p in trainable_params),
    )

    # v1.02: 检测 Kohya 风格目录结构
    use_kohya_repeat = detect_kohya_structure(args.data_dir)
    if use_kohya_repeat:
        logger.info("Detected Kohya-style directory structure, using per-sample repeat")

    # 创建数据集
    base_dataset = CaptionDataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        enable_bucket=True,
        min_bucket_reso=args.min_reso,
        max_bucket_reso=args.max_reso,
        bucket_reso_steps=args.reso_step,
        max_bucket_aspect_ratio=args.max_ar,
        shuffle_caption=args.shuffle_caption,
        keep_tokens=args.keep_tokens,
        flip_augment=args.flip_augment,
    )
    if bool(args.shuffle_caption_per_epoch):
        logger.warning("shuffle_caption_per_epoch is ignored; using Kohya-style per-step caption shuffle.")
    base_dataset.shuffle_caption_per_epoch = False

    # v1.02: Latent 缓存
    latent_cache = None
    latent_cache_stats = None
    if args.cache_latents and vae is not None:
        cache_dir = args.cache_dir or str(Path(args.data_dir) / ".latent_cache")
        cache_namespace, cache_meta = build_latent_cache_namespace(args)
        latent_cache = LatentCache(
            cache_dir,
            vae,
            device,
            dtype,
            cache_namespace=cache_namespace,
            cache_meta=cache_meta,
        )
        logger.info("Latent cache namespace: %s", cache_namespace)
        logger.info("Latent cache path: %s", latent_cache.cache_dir)
        latent_cache_stats = cache_all_latents(base_dataset, vae, latent_cache, device, dtype, logger)
        if latent_cache_stats.get("complete", False):
            # If everything is cached, we can free VRAM by offloading the VAE.
            logger.info("All latents cached; offloading VAE to CPU to save VRAM.")
            try:
                vae.model.to("cpu")
                latent_cache.vae = None
            except Exception as exc:
                logger.warning("Failed to offload VAE to CPU: %s", exc)
            vae = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    # v1.02: 根据目录结构选择数据集包装器
    working_dataset = base_dataset
    if latent_cache is not None:
        working_dataset = LatentCacheDataset(base_dataset, latent_cache)

    if use_kohya_repeat:
        dataset = KohyaRepeatDataset(working_dataset)
    elif args.repeats > 1:
        dataset = RepeatDataset(working_dataset, args.repeats)
    else:
        dataset = working_dataset

    if len(dataset) == 0:
        logger.error("No samples found in %s", args.data_dir)
        return

    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        seed=args.seed,
    )
    logger.info("Dataset: %d samples, %d batches per epoch", len(dataset), len(dataloader))

    # 优化器
    optimizer = build_optimizer(args, trainable_params, logger)
    logger.info("Optimizer: %s (lr=%.3e, wd=%.3e)", getattr(args, "optimizer", "adamw"), float(args.lr), float(getattr(args, "weight_decay", 0.0)))

    # Scaler (needed for resume)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None

    # Resume (after LoRA injection + optimizer creation)
    start_epoch = 0
    global_step = 0
    losses = []
    if args.resume:
        logger.info("Resuming from training state: %s", args.resume)
        state = load_training_state(args.resume)

        snap = state.get("args_snapshot", {}) or {}
        ckpt_network = str(snap.get("network_type", "lora") or "lora").lower()
        cur_network = str(getattr(args, "network_type", "lora") or "lora").lower()
        if ckpt_network != cur_network:
            raise RuntimeError(f"Resume mismatch: network_type differs (ckpt={ckpt_network}, current={cur_network})")
        if int(snap.get("lora_rank", args.lora_rank)) != int(args.lora_rank):
            raise RuntimeError("Resume mismatch: lora_rank differs from checkpoint")
        if float(snap.get("lora_alpha", args.lora_alpha)) != float(args.lora_alpha):
            raise RuntimeError("Resume mismatch: lora_alpha differs from checkpoint")
        if cur_network == "lokr":
            if int(snap.get("lokr_factor", getattr(args, "lokr_factor", 8))) != int(getattr(args, "lokr_factor", 8)):
                raise RuntimeError("Resume mismatch: lokr_factor differs from checkpoint")
            if bool(snap.get("lokr_full_matrix", getattr(args, "lokr_full_matrix", False))) != bool(getattr(args, "lokr_full_matrix", False)):
                raise RuntimeError("Resume mismatch: lokr_full_matrix differs from checkpoint")
            if bool(snap.get("lokr_decompose_both", getattr(args, "lokr_decompose_both", False))) != bool(getattr(args, "lokr_decompose_both", False)):
                raise RuntimeError("Resume mismatch: lokr_decompose_both differs from checkpoint")
            if bool(snap.get("lokr_use_tucker", getattr(args, "lokr_use_tucker", False))) != bool(getattr(args, "lokr_use_tucker", False)):
                raise RuntimeError("Resume mismatch: lokr_use_tucker differs from checkpoint")
        if str(snap.get("mixed_precision", args.mixed_precision)) != str(args.mixed_precision):
            logger.warning("Resume warning: mixed_precision differs (ckpt=%s current=%s)",
                           snap.get("mixed_precision"), args.mixed_precision)
        if str(snap.get("optimizer", getattr(args, "optimizer", "adamw"))).lower() != str(
            getattr(args, "optimizer", "adamw")
        ).lower():
            raise RuntimeError("Resume mismatch: optimizer type differs from checkpoint")

        lora_injector.load_state_dict(state["lora_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        _optimizer_state_to_device(optimizer, device)

        if scaler is not None and state.get("scaler_state") is not None:
            try:
                scaler.load_state_dict(state["scaler_state"])
            except Exception as exc:
                logger.warning("Could not restore GradScaler state: %s", exc)

        start_epoch = int(state.get("next_epoch", 0))
        global_step = int(state.get("global_step", 0))
        losses = list(state.get("losses_tail", []) or [])
        _set_rng_state(state.get("rng_state"))
        logger.info("Resume loaded: next_epoch=%d global_step=%d", start_epoch, global_step)

    # 计算总步数
    import math
    steps_per_epoch = int(math.ceil(len(dataloader) / max(1, args.grad_accum)))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # wandb init (after args resolved, before training loop)
    try:
        wandb = _maybe_init_wandb(args, logger)
        if wandb is not None:
            wandb_run = wandb.run
    except Exception as exc:
        logger.warning("wandb init failed (continuing without): %s", exc)
        wandb = None
        wandb_run = None

    # 进度显示
    show_progress = not args.no_progress
    progress, task_id, progress_type = init_progress(show_progress, total_steps)
    if int(getattr(args, "text_cache_size", 0) or 0) > 0:
        logger.info("text_cache_size is deprecated and ignored (LRU cache has been removed).")

    import time
    start_time = time.time()
    monitor = TrainingMonitor(
        args=args,
        logger=logger,
        wandb_obj=wandb,
        total_steps=total_steps,
        start_time=start_time,
        device=device,
    )
    if int(getattr(args, "wandb_log_every", 1) or 1) != int(getattr(args, "monitor_every", 5) or 5):
        logger.info(
            "wandb_log_every=%d is deprecated in unified monitor flow; monitor_every=%d is used.",
            int(getattr(args, "wandb_log_every", 1) or 1),
            int(getattr(args, "monitor_every", 5) or 5),
        )

    # Fail fast if adapter gradients are silently zero in early updates.
    grad_watch_checks = 0 if global_step > 0 else 3

    if progress_type == "rich":
        progress.start()
        if global_step > 0:
            progress.update(task_id, completed=min(global_step, total_steps), loss=0.0, lr=args.lr, speed=0.0)

    current_epoch_for_crash = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch_for_crash = epoch
            logger.info("Epoch %d/%d", epoch + 1, args.epochs)
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_count = 0
            accum_samples = 0
            step_in_epoch = 0
            try:
                # Update dataset epoch for deterministic caption shuffle if enabled.
                if hasattr(base_dataset, "set_epoch"):
                    base_dataset.set_epoch(epoch, seed=args.seed)
            except Exception:
                pass

            for batch_idx, batch in enumerate(dataloader):
                captions = batch["captions"]

                # 编码图像 (支持 latent 缓存)
                with torch.no_grad():
                    if latent_cache is not None and "latent" in batch:
                        # Prefer cached latents and avoid moving images to GPU when possible.
                        use_flags = batch.get("use_cached_latent", [False] * len(batch["latent"]))
                        if all(use_flags):
                            latents = torch.stack(batch["latent"]).to(device, dtype=dtype)
                        else:
                            if vae is None:
                                raise RuntimeError(
                                    "Latent cache is incomplete but VAE is not available. "
                                    "Disable VAE offload or pre-cache all latents."
                                )
                            images = batch["images"].to(device, dtype=dtype)
                            latents = []
                            for i in range(len(use_flags)):
                                if use_flags[i]:
                                    latents.append(batch["latent"][i].to(device, dtype=dtype))
                                else:
                                    img_5d = images[i : i + 1].unsqueeze(2)
                                    latents.append(vae.model.encode(img_5d, vae.scale).squeeze(0))
                            latents = torch.stack(latents, dim=0)
                    elif vae is not None:
                        images = batch["images"].to(device, dtype=dtype)
                        images_5d = images.unsqueeze(2)
                        latents = vae.model.encode(images_5d, vae.scale)
                    else:
                        images = batch["images"].to(device, dtype=dtype)
                        latents = images.unsqueeze(2)

                bs = latents.shape[0]

                # 编码文本：Qwen embeddings；如 T5 tokenizer 可用则通过 llm_adapter 进一步处理。
                with torch.no_grad():
                    if qwen_tokenizer and qwen_model:
                        toks = tokenize_qwen(qwen_tokenizer, captions, args.seq_len)
                        qwen_input_ids = toks["input_ids"]
                        qwen_attention_mask = toks["attention_mask"]

                        t5_ids = None
                        if t5_tokenizer is not None:
                            t5_toks = tokenize_t5(t5_tokenizer, captions, max_length=args.seq_len)
                            t5_ids = t5_toks["input_ids"].to(device)

                        qwen_embeds = compute_qwen_embeddings(qwen_model, qwen_input_ids, qwen_attention_mask)
                        qwen_embeds = qwen_embeds.to(device=device, dtype=dtype)

                        cross = model.preprocess_text_embeds(qwen_embeds, t5_ids)
                        if cross.shape[1] < args.seq_len:
                            cross = F.pad(cross, (0, 0, 0, args.seq_len - cross.shape[1]))
                        elif cross.shape[1] > args.seq_len:
                            cross = cross[:, : args.seq_len]
                    else:
                        # Fallback: keep shape consistent, but training quality will degrade without proper encoders.
                        cross = torch.zeros(bs, args.seq_len, 1024, device=device, dtype=dtype)

                # 采样时间步和噪声
                t = sample_t(bs, device)
                t = t.to(dtype=latents.dtype)
                noise = torch.randn_like(latents)
                noisy_latents = (1 - t.view(-1, 1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1, 1) * noise

                # padding mask
                # Match v1.01: 4D mask (B,1,H,W) filled with zeros (no padding).
                padding_mask = torch.zeros(
                    (bs, 1, noisy_latents.shape[-2], noisy_latents.shape[-1]),
                    device=device,
                    dtype=noisy_latents.dtype,
                )

                # 前向传播
                with torch.amp.autocast("cuda", dtype=dtype):
                    pred = forward_with_optional_checkpoint(
                        model, noisy_latents, t, cross, padding_mask, args.grad_checkpoint
                    )
                    target = noise - latents
                    # Match v1.01: compute loss in fp32 for stability.
                    loss = F.mse_loss(pred.float(), target.float())

                # 反向传播
                loss_scaled = loss / args.grad_accum
                if scaler:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                accum_loss += loss.item()
                accum_count += 1
                accum_samples += int(bs)

                # 梯度累积完成后更新
                if (batch_idx + 1) % args.grad_accum == 0:
                    grad_watch_checks = _check_adapter_gradients(
                        trainable_params,
                        logger,
                        step_hint=global_step + 1,
                        remaining_checks=grad_watch_checks,
                    )
                    grad_nonzero_ratio = _compute_grad_nonzero_ratio(trainable_params)
                    _maybe_clip_grads(args, scaler=scaler, optimizer=optimizer, trainable_params=trainable_params)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    step_in_epoch += 1
                    avg_loss = accum_loss / max(1, accum_count)
                    losses.append(avg_loss)
                    update_samples = max(1, int(accum_samples))
                    accum_loss = 0.0
                    accum_count = 0
                    accum_samples = 0

                    # 计算速度
                    elapsed = time.time() - start_time
                    speed = global_step / elapsed if elapsed > 0 else 0
                    samples_per_s = float(speed) * float(update_samples)
                    current_lr = _get_current_lr(optimizer, float(args.lr))
                    prodigy_d = _get_prodigy_d(optimizer)

                    # 更新进度
                    if progress_type == "rich":
                        progress.update(task_id, completed=global_step, loss=avg_loss, lr=current_lr, speed=speed)
                    elif global_step % args.log_every == 0:
                        logger.info("Step %d/%d, loss=%.4f, speed=%.3f it/s",
                                    global_step, total_steps, avg_loss, speed)
                    monitor.log_step_if_needed(
                        epoch=epoch + 1,
                        global_step=global_step,
                        step_in_epoch=step_in_epoch,
                        loss=avg_loss,
                        lr=current_lr,
                        speed_it_s=speed,
                        samples_per_s=samples_per_s,
                        grad_accum=int(args.grad_accum),
                        optimizer_name=str(getattr(args, "optimizer", "adamw") or "adamw"),
                        prodigy_d=prodigy_d,
                        grad_nonzero_ratio=grad_nonzero_ratio,
                    )

                    # 保存检查点
                    if (not args.save_every_epoch) and args.save_every > 0 and global_step % args.save_every == 0:
                        ckpt_path = output_dir / f"{args.output_name}_step{global_step}.safetensors"
                        if args.comfyui_format:
                            lora_injector.save_comfyui(str(ckpt_path))
                        else:
                            lora_injector.save(str(ckpt_path))
                        logger.info("Saved checkpoint: %s", ckpt_path)
                        if args.save_state and int(getattr(args, "save_state_every", 0) or 0) <= 0:
                            state_path = output_dir / f"{args.output_name}_step{global_step}.state.pt"
                            save_training_state(
                                str(state_path),
                                lora_injector=lora_injector,
                                optimizer=optimizer,
                                scaler=scaler,
                                next_epoch=epoch,
                                global_step=global_step,
                                losses=losses,
                                args=args,
                            )
                            logger.info("Saved training state: %s", state_path)

                    # (disabled) rich loss curve display: use wandb instead.

                    # 检查最大步数
                    if args.max_steps > 0 and global_step >= args.max_steps:
                        logger.info("Reached max_steps=%d", args.max_steps)
                        break

            # Flush remainder grads (when batches_per_epoch is not divisible by grad_accum)
            if accum_count > 0 and (args.max_steps <= 0 or global_step < args.max_steps):
                # We scaled loss by 1/grad_accum each backward. For a partial accumulation, rescale grads back
                # to match averaging by `accum_count`.
                factor = float(args.grad_accum) / float(accum_count)
                if scaler:
                    try:
                        scaler.unscale_(optimizer)
                    except Exception:
                        pass
                for group in optimizer.param_groups:
                    for p in group.get("params", []):
                        if p is not None and getattr(p, "grad", None) is not None:
                            p.grad.mul_(factor)

                grad_watch_checks = _check_adapter_gradients(
                    trainable_params,
                    logger,
                    step_hint=global_step + 1,
                    remaining_checks=grad_watch_checks,
                )
                grad_nonzero_ratio = _compute_grad_nonzero_ratio(trainable_params)
                _maybe_clip_grads(args, scaler=scaler, optimizer=optimizer, trainable_params=trainable_params)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                step_in_epoch += 1
                avg_loss = accum_loss / max(1, accum_count)
                losses.append(avg_loss)
                update_samples = max(1, int(accum_samples))
                accum_loss = 0.0
                accum_count = 0
                accum_samples = 0

                elapsed = time.time() - start_time
                speed = global_step / elapsed if elapsed > 0 else 0
                samples_per_s = float(speed) * float(update_samples)
                current_lr = _get_current_lr(optimizer, float(args.lr))
                prodigy_d = _get_prodigy_d(optimizer)
                if progress_type == "rich":
                    progress.update(task_id, completed=global_step, loss=avg_loss, lr=current_lr, speed=speed)
                elif global_step % args.log_every == 0:
                    logger.info("Step %d/%d, loss=%.4f, speed=%.3f it/s",
                                global_step, total_steps, avg_loss, speed)
                monitor.log_step_if_needed(
                    epoch=epoch + 1,
                    global_step=global_step,
                    step_in_epoch=step_in_epoch,
                    loss=avg_loss,
                    lr=current_lr,
                    speed_it_s=speed,
                    samples_per_s=samples_per_s,
                    grad_accum=int(args.grad_accum),
                    optimizer_name=str(getattr(args, "optimizer", "adamw") or "adamw"),
                    prodigy_d=prodigy_d,
                    grad_nonzero_ratio=grad_nonzero_ratio,
                )

            # epoch 结束后保存
            if args.save_every_epoch and (max(1, int(args.save_every or 1)) > 0) and (
                (epoch + 1) % max(1, int(args.save_every or 1)) == 0
            ):
                ckpt_path = output_dir / f"{args.output_name}_epoch{epoch + 1}.safetensors"
                if args.comfyui_format:
                    lora_injector.save_comfyui(str(ckpt_path))
                else:
                    lora_injector.save(str(ckpt_path))
                logger.info("Saved epoch checkpoint: %s", ckpt_path)
                if args.save_state and int(getattr(args, "save_state_every", 0) or 0) <= 0:
                    state_path = output_dir / f"{args.output_name}_epoch{epoch + 1}.state.pt"
                    save_training_state(
                        str(state_path),
                        lora_injector=lora_injector,
                        optimizer=optimizer,
                        scaler=scaler,
                        next_epoch=epoch + 1,
                        global_step=global_step,
                        losses=losses,
                        args=args,
                    )
                    logger.info("Saved training state: %s", state_path)

            # Save state-only checkpoint (optional, epoch-based)
            if args.save_state and args.save_state_every > 0 and (epoch + 1) % args.save_state_every == 0:
                state_path = output_dir / f"{args.output_name}_state_epoch{epoch + 1}.state.pt"
                save_training_state(
                    str(state_path),
                    lora_injector=lora_injector,
                    optimizer=optimizer,
                    scaler=scaler,
                    next_epoch=epoch + 1,
                    global_step=global_step,
                    losses=losses,
                    args=args,
                )
                logger.info("Saved training state: %s", state_path)

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

    except Exception as exc:
        logger.error("Training crashed: %s", exc)
        if args.save_state and global_step > 0:
            try:
                crash_path = output_dir / f"{args.output_name}_crash_step{global_step}.state.pt"
                save_training_state(
                    str(crash_path),
                    lora_injector=lora_injector,
                    optimizer=optimizer,
                    scaler=scaler,
                    next_epoch=current_epoch_for_crash,
                    global_step=global_step,
                    losses=losses,
                    args=args,
                )
                logger.info("Saved crash recovery state: %s", crash_path)
            except Exception as exc2:
                logger.warning("Failed to save crash recovery state: %s", exc2)
        raise
    finally:
        if progress_type == "rich":
            progress.stop()
        if wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass

    # 最终保存
    final_path = output_dir / f"{args.output_name}.safetensors"
    if args.comfyui_format:
        lora_injector.save_comfyui(str(final_path))
        logger.info("Saved final LoRA (ComfyUI format): %s", final_path)
    else:
        lora_injector.save(str(final_path))
        logger.info("Saved final LoRA: %s", final_path)
    if args.save_state:
        final_state = output_dir / f"{args.output_name}.state.pt"
        save_training_state(
            str(final_state),
            lora_injector=lora_injector,
            optimizer=optimizer,
            scaler=scaler,
            next_epoch=args.epochs,
            global_step=global_step,
            losses=losses,
            args=args,
        )
        logger.info("Saved final training state: %s", final_state)

    logger.info("Training complete! Total steps: %d", global_step)


if __name__ == "__main__":
    main()
