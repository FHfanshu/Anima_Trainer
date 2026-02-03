from __future__ import annotations

from typing import Any, Dict


def normalize_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a training config payload into the flat schema expected by `train.py`.

    The GUI historically used a kohya-style nested schema (e.g. `network.*`, `optimizer.*`).
    The backend training script uses a flat schema (e.g. `lora_type`, `learning_rate`).
    This function makes the API tolerant and keeps stored configs consistent.
    """
    cfg: Dict[str, Any] = dict(config or {})

    # Dataset path
    if not cfg.get("data_root") and cfg.get("train_data_dir"):
        cfg["data_root"] = cfg["train_data_dir"]

    # Resume
    if not cfg.get("resume_from_checkpoint") and cfg.get("resume"):
        cfg["resume_from_checkpoint"] = cfg["resume"]

    # Network / LoRA
    network = cfg.get("network")
    if isinstance(network, dict):
        if not cfg.get("lora_type") and network.get("network_type"):
            cfg["lora_type"] = network.get("network_type")
        if cfg.get("lora_rank") is None and network.get("network_dim") is not None:
            cfg["lora_rank"] = network.get("network_dim")
        if cfg.get("lora_alpha") is None and network.get("network_alpha") is not None:
            cfg["lora_alpha"] = network.get("network_alpha")
        if cfg.get("lora_dropout") is None and network.get("network_dropout") is not None:
            cfg["lora_dropout"] = network.get("network_dropout")

    # Optimizer block (nested) -> flat
    optimizer = cfg.get("optimizer")
    if isinstance(optimizer, dict):
        opt_type = optimizer.get("optimizer_type")
        if isinstance(opt_type, str):
            opt_type_lower = opt_type.lower()
            if opt_type_lower in ("adamw8bit", "adamw_8bit", "adamw 8bit"):
                cfg["optimizer"] = "adamw8bit"
            elif opt_type_lower in ("adamw",):
                cfg["optimizer"] = "adamw"

        if cfg.get("learning_rate") is None and optimizer.get("learning_rate") is not None:
            cfg["learning_rate"] = optimizer.get("learning_rate")
        if not cfg.get("lr_scheduler") and optimizer.get("lr_scheduler"):
            cfg["lr_scheduler"] = optimizer.get("lr_scheduler")
        if cfg.get("lr_warmup_steps") is None and optimizer.get("lr_warmup_steps") is not None:
            cfg["lr_warmup_steps"] = optimizer.get("lr_warmup_steps")
        if cfg.get("lr_num_cycles") is None and optimizer.get("lr_scheduler_num_cycles") is not None:
            cfg["lr_num_cycles"] = optimizer.get("lr_scheduler_num_cycles")
        if cfg.get("lr_power") is None and optimizer.get("lr_scheduler_power") is not None:
            cfg["lr_power"] = optimizer.get("lr_scheduler_power")
        if cfg.get("max_grad_norm") is None and optimizer.get("max_grad_norm") is not None:
            cfg["max_grad_norm"] = optimizer.get("max_grad_norm")

    # Mixed precision: backend supports no/fp16/bf16
    mixed_precision = cfg.get("mixed_precision")
    if mixed_precision is not None:
        mp = str(mixed_precision)
        if mp not in ("no", "fp16", "bf16"):
            cfg["mixed_precision"] = "bf16"

    return cfg

