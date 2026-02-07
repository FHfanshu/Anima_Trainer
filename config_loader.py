"""
TOML 配置加载模块 for Anima Trainer v1.02

支持从 TOML 文件加载训练配置，并映射到 argparse 参数。
"""

import tomllib
from pathlib import Path
from typing import Any


def load_toml_config(config_path: str) -> dict:
    """
    加载 TOML 配置文件。

    Args:
        config_path: TOML 配置文件路径

    Returns:
        解析后的配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    return config


def apply_config_to_args(args, config: dict) -> None:
    """
    将 TOML 配置映射到 argparse 参数对象。

    命令行参数优先级高于配置文件（如果命令行显式指定了值）。

    Args:
        args: argparse.Namespace 对象
        config: TOML 配置字典
    """
    # 模型路径映射
    model_config = config.get("model", {})
    _set_if_default(args, "transformer", model_config.get("transformer_path", ""), default_value="")
    _set_if_default(args, "vae", model_config.get("vae_path", ""), default_value="")
    _set_if_default(args, "qwen", model_config.get("text_encoder_path", ""), default_value="")
    _set_if_default(args, "t5_tokenizer_dir", model_config.get("t5_tokenizer_path", ""), default_value="")

    # 数据集配置映射
    dataset_config = config.get("dataset", {})
    _set_if_default(args, "data_dir", dataset_config.get("data_dir", ""), default_value="")
    _set_if_default(args, "resolution", dataset_config.get("resolution", 1024), default_value=1024)
    _set_if_default(args, "min_reso", dataset_config.get("min_reso", 512), default_value=512)
    _set_if_default(args, "max_reso", dataset_config.get("max_reso", 2048), default_value=2048)
    _set_if_default(args, "reso_step", dataset_config.get("reso_step", 64), default_value=64)
    _set_if_default(args, "max_ar", dataset_config.get("max_ar", 2.0), default_value=2.0)
    _set_if_default(args, "repeats", dataset_config.get("repeats", 1), default_value=1)
    _set_if_default(args, "shuffle_caption", dataset_config.get("shuffle_caption", False), default_value=False)
    _set_if_default(
        args,
        "shuffle_caption_per_epoch",
        dataset_config.get("shuffle_caption_per_epoch", False),
        default_value=False,
    )
    _set_if_default(args, "keep_tokens", dataset_config.get("keep_tokens", 0), default_value=0)
    _set_if_default(args, "flip_augment", dataset_config.get("flip_augment", False), default_value=False)
    _set_if_default(args, "cache_latents", dataset_config.get("cache_latents", False), default_value=False)
    _set_if_default(args, "cache_dir", dataset_config.get("cache_dir", ""), default_value="")

    # LoRA 配置映射
    lora_config = config.get("lora", {})
    _set_if_default(args, "lora_rank", lora_config.get("lora_rank", 32), default_value=32)
    if "lora_alpha" in lora_config and lora_config.get("lora_alpha") is not None:
        lora_alpha_val = float(lora_config.get("lora_alpha"))
    else:
        lora_alpha_val = 16.0
    _set_if_default(args, "lora_alpha", lora_alpha_val, default_value=16.0)
    _set_if_default(args, "lora_dropout", lora_config.get("lora_dropout", 0.0), default_value=0.0)
    _set_if_default(args, "lora_targets", lora_config.get("lora_targets", ""), default_value="")
    _set_if_default(args, "anima_target_preset", lora_config.get("anima_target_preset", "v101"), default_value="v101")
    _set_if_default(args, "network_type", lora_config.get("network_type", "lora"), default_value="lora")
    _set_if_default(args, "lokr_factor", lora_config.get("lokr_factor", 8), default_value=8)
    _set_if_default(args, "lokr_use_tucker", lora_config.get("lokr_use_tucker", False), default_value=False)
    _set_if_default(args, "lokr_decompose_both", lora_config.get("lokr_decompose_both", False), default_value=False)
    _set_if_default(args, "lokr_full_matrix", lora_config.get("lokr_full_matrix", False), default_value=False)
    _set_if_default(args, "lokr_rank_dropout", lora_config.get("lokr_rank_dropout", 0.0), default_value=0.0)
    _set_if_default(args, "lokr_module_dropout", lora_config.get("lokr_module_dropout", 0.0), default_value=0.0)
    _set_if_default(args, "lokr_constraint", lora_config.get("lokr_constraint", 0.0), default_value=0.0)
    _set_if_default(args, "lokr_normalize", lora_config.get("lokr_normalize", False), default_value=False)
    # lora_name 用于输出名称
    if lora_config.get("lora_name"):
        _set_if_default(args, "output_name", lora_config.get("lora_name"), default_value="anima_lora")

    # 训练参数映射
    training_config = config.get("training", {})
    _set_if_default(args, "epochs", training_config.get("epochs", 1), default_value=1)
    _set_if_default(args, "max_steps", training_config.get("max_steps", 0), default_value=0)
    _set_if_default(args, "batch_size", training_config.get("batch_size", 1), default_value=1)
    _set_if_default(args, "grad_accum", training_config.get("grad_accum", 1), default_value=1)
    _set_if_default(args, "grad_clip", training_config.get("grad_clip", 0.0), default_value=0.0)
    _set_if_default(args, "lr", training_config.get("learning_rate", 5e-5), default_value=5e-5)
    _set_if_default(args, "mixed_precision", training_config.get("mixed_precision", "bf16"), default_value="bf16")
    _set_if_default(args, "grad_checkpoint", training_config.get("grad_checkpoint", False), default_value=False)
    _set_if_default(args, "num_workers", training_config.get("num_workers", 0), default_value=0)
    _set_if_default(args, "seed", training_config.get("seed", 42), default_value=42)
    _set_if_default(args, "xformers", training_config.get("xformers", False), default_value=False)
    _set_if_default(args, "resume", training_config.get("resume", ""), default_value="")
    _set_if_default(args, "text_cache_size", training_config.get("text_cache_size", 256), default_value=256)
    _set_if_default(args, "seq_len", training_config.get("seq_len", 512), default_value=512)
    _set_if_default(args, "log_every", training_config.get("log_every", 10), default_value=10)
    _set_if_default(args, "auto_install", training_config.get("auto_install", False), default_value=False)
    _set_if_default(args, "interactive", training_config.get("interactive", False), default_value=False)

    # 优化器配置映射（可选）
    optimizer_config = config.get("optimizer", {})
    _set_if_default(args, "optimizer", optimizer_config.get("type", "adamw"), default_value="adamw")
    _set_if_default(args, "weight_decay", optimizer_config.get("weight_decay", 0.01), default_value=0.01)
    _set_if_default(args, "beta1", optimizer_config.get("beta1", 0.9), default_value=0.9)
    _set_if_default(args, "beta2", optimizer_config.get("beta2", 0.999), default_value=0.999)
    _set_if_default(args, "eps", optimizer_config.get("eps", 1e-8), default_value=1e-8)
    _set_if_default(args, "prodigy_beta3", optimizer_config.get("prodigy_beta3", 0.999), default_value=0.999)
    _set_if_default(args, "prodigy_d0", optimizer_config.get("prodigy_d0", 1e-6), default_value=1e-6)
    _set_if_default(args, "prodigy_d_coef", optimizer_config.get("prodigy_d_coef", 1.0), default_value=1.0)
    _set_if_default(args, "prodigy_growth_rate", optimizer_config.get("prodigy_growth_rate", float("inf")), default_value=float("inf"))
    _set_if_default(args, "prodigy_slice_p", optimizer_config.get("prodigy_slice_p", 1), default_value=1)
    _set_if_default(args, "prodigy_decouple", optimizer_config.get("prodigy_decouple", True), default_value=True)
    _set_if_default(
        args,
        "prodigy_use_bias_correction",
        optimizer_config.get("prodigy_use_bias_correction", True),
        default_value=True,
    )
    _set_if_default(
        args,
        "prodigy_safeguard_warmup",
        optimizer_config.get("prodigy_safeguard_warmup", True),
        default_value=True,
    )

    # 输出配置映射
    output_config = config.get("output", {})
    _set_if_default(args, "output_dir", output_config.get("output_dir", "./output"), default_value="./output")
    _set_if_default(args, "output_name", output_config.get("output_name", "anima_lora"), default_value="anima_lora")
    # save_every 在 TOML 中是 epoch 单位（每 N 个 epoch 保存一次）
    save_every = output_config.get("save_every", 0)
    if save_every > 0:
        _set_if_default(args, "save_every", int(save_every), default_value=0)
        _set_if_default(args, "save_every_epoch", True, default_value=False)
    _set_if_default(args, "save_state", output_config.get("save_state", True), default_value=True)
    _set_if_default(args, "save_state_every", output_config.get("save_state_every", 0), default_value=0)
    _set_if_default(args, "comfyui_format", output_config.get("comfyui_format", False), default_value=False)

    # 进度显示配置
    progress_config = config.get("progress", {})
    _set_if_default(args, "loss_curve_steps", progress_config.get("loss_curve_steps", 100), default_value=100)
    _set_if_default(args, "no_progress", progress_config.get("no_progress", False), default_value=False)
    _set_if_default(args, "no_live_curve", progress_config.get("no_live_curve", True), default_value=False)

    # 训练监控配置（可选）
    monitor_config = config.get("monitor", {})
    _set_if_default(args, "monitor_enabled", monitor_config.get("enabled", True), default_value=True)
    _set_if_default(args, "monitor_every", monitor_config.get("every", 5), default_value=5)
    _set_if_default(args, "monitor_memory", monitor_config.get("memory", True), default_value=True)
    _set_if_default(args, "monitor_wandb", monitor_config.get("wandb", True), default_value=True)
    _set_if_default(args, "monitor_alert_policy", monitor_config.get("alert_policy", "warn"), default_value="warn")

    # wandb 配置映射（可选）
    wandb_config = config.get("wandb", {})
    _set_if_default(args, "wandb", wandb_config.get("enabled", False), default_value=False)
    _set_if_default(args, "wandb_project", wandb_config.get("project", ""), default_value="")
    _set_if_default(args, "wandb_entity", wandb_config.get("entity", ""), default_value="")
    _set_if_default(args, "wandb_name", wandb_config.get("name", ""), default_value="")
    _set_if_default(args, "wandb_group", wandb_config.get("group", ""), default_value="")
    tags = wandb_config.get("tags", "")
    if isinstance(tags, list):
        tags = ",".join([str(x) for x in tags])
    _set_if_default(args, "wandb_tags", tags, default_value="")
    _set_if_default(args, "wandb_mode", wandb_config.get("mode", "online"), default_value="online")
    _set_if_default(args, "wandb_log_every", wandb_config.get("log_every", 1), default_value=1)


def _set_if_default(args, attr: str, value: Any, default_value: Any = None) -> None:
    """
    仅当 args 中的属性为默认值时才设置新值。

    这确保命令行参数优先级高于配置文件。
    """
    if not hasattr(args, attr):
        setattr(args, attr, value)
        return

    current = getattr(args, attr)

    # 当给定 default_value 时，只在当前值仍是解析器默认值时覆盖，保证 CLI 优先级。
    if default_value is not None:
        if current == default_value:
            setattr(args, attr, value)
        return

    if current in ("", None, 0, False, []):
        setattr(args, attr, value)
