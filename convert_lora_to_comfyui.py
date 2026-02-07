#!/usr/bin/env python
"""
将 Anima 训练器输出的 LoRA 转换为 ComfyUI 兼容格式

用法:
    python convert_lora_to_comfyui.py input.safetensors output.safetensors
    python convert_lora_to_comfyui.py input.safetensors  # 输出到 input_comfyui.safetensors
"""

import argparse
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# 已知的层名模式，用于正确分割下划线
KNOWN_PATTERNS = [
    "llm_adapter",
    "blocks",
    "self_attn",
    "cross_attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "output_proj",
    "o_proj",
    "mlp",
    "layer1",
    "layer2",
    "adaln_modulation_self_attn",
    "adaln_modulation_cross_attn",
    "adaln_modulation_mlp",
    "layer_norm_self_attn",
    "layer_norm_cross_attn",
    "layer_norm_mlp",
]


def convert_key(old_key: str) -> str:
    """
    转换键名格式:
    transformer_blocks_0_self_attn_q_proj.lora_down.weight
    -> diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight
    """
    if old_key.startswith("diffusion_model.") and ".lora_" in old_key:
        return old_key.replace("diffusion_model.llm.adapter.", "diffusion_model.llm_adapter.")

    # 分离 .lora_down.weight 或 .lora_up.weight 后缀
    if ".lora_down.weight" in old_key:
        base = old_key.replace(".lora_down.weight", "")
        suffix = ".lora_down.weight"
    elif ".lora_up.weight" in old_key:
        base = old_key.replace(".lora_up.weight", "")
        suffix = ".lora_up.weight"
    else:
        return old_key  # 未知格式，保持原样

    # 移除 transformer_ 前缀
    if base.startswith("transformer_"):
        base = base[len("transformer_"):]

    # 智能分割：识别已知模式和数字
    # blocks_0_self_attn_q_proj -> blocks.0.self_attn.q_proj
    result_parts = []
    remaining = base

    while remaining:
        matched = False

        # 先检查是否以数字开头
        num_match = re.match(r'^(\d+)_?', remaining)
        if num_match:
            result_parts.append(num_match.group(1))
            remaining = remaining[len(num_match.group(0)):]
            matched = True
            continue

        # 检查已知模式（按长度降序，优先匹配更长的）
        for pattern in sorted(KNOWN_PATTERNS, key=len, reverse=True):
            if remaining.startswith(pattern):
                result_parts.append(pattern)
                remaining = remaining[len(pattern):]
                if remaining.startswith("_"):
                    remaining = remaining[1:]
                matched = True
                break

        # 如果没有匹配到已知模式，取到下一个下划线
        if not matched:
            idx = remaining.find("_")
            if idx == -1:
                result_parts.append(remaining)
                remaining = ""
            else:
                result_parts.append(remaining[:idx])
                remaining = remaining[idx + 1:]

    # 重新组合，用点分隔
    new_base = ".".join(result_parts)

    # 添加 diffusion_model 前缀
    new_key = f"diffusion_model.{new_base}{suffix}"
    return new_key.replace("diffusion_model.llm.adapter.", "diffusion_model.llm_adapter.")


def load_lora(path: str) -> tuple[dict, dict]:
    """加载 LoRA 文件，返回权重和元数据"""
    weights = {}
    metadata = {}

    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = dict(f.metadata()) if f.metadata() else {}
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    return weights, metadata


def convert_lora(input_path: str, output_path: str = None):
    """转换 LoRA 文件"""
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_stem(f"{input_path.stem}_comfyui")
    else:
        output_path = Path(output_path)

    print(f"加载: {input_path}")
    weights, metadata = load_lora(str(input_path))

    # 获取 rank 和 alpha
    rank = int(metadata.get("ss_network_dim", "32"))
    alpha = float(metadata.get("ss_network_alpha", str(rank)))

    print(f"LoRA rank: {rank}, alpha: {alpha}")
    print(f"原始键数量: {len(weights)}")

    # 转换键名
    new_weights = {}
    converted_layers = set()

    for old_key, tensor in weights.items():
        new_key = convert_key(old_key)
        if new_key in new_weights:
            raise ValueError(f"转换后出现键冲突: {new_key}")
        new_weights[new_key] = tensor

        # 仅为 LoRA 权重层记录 alpha，避免覆盖非 LoRA 键
        if new_key.endswith(".lora_down.weight") or new_key.endswith(".lora_up.weight"):
            layer_name = new_key.rsplit(".lora_", 1)[0]
            converted_layers.add(layer_name)

        if old_key != new_key:
            print(f"  {old_key}")
            print(f"    -> {new_key}")

    # 为每个层添加 alpha
    for layer_name in converted_layers:
        alpha_key = f"{layer_name}.alpha"
        if alpha_key in new_weights:
            continue
        new_weights[alpha_key] = torch.tensor(float(alpha), dtype=torch.float32)

    print(f"\n转换后键数量: {len(new_weights)}")
    print(f"  - LoRA 权重: {len(converted_layers) * 2}")
    print(f"  - Alpha 值: {len(converted_layers)}")

    # 更新元数据
    new_metadata = dict(metadata)
    new_metadata.update({
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(alpha),
        "ss_output_name": metadata.get("ss_output_name", "anima_lora"),
        "format": "comfyui_compatible",
    })

    # 保存
    save_file(new_weights, str(output_path), metadata=new_metadata)
    print(f"\n保存到: {output_path}")


def batch_convert(input_dir: str, output_dir: str = None, pattern: str = "*.safetensors"):
    """批量转换目录下的所有 LoRA 文件"""
    import glob

    input_dir = Path(input_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_dir

    files = list(input_dir.glob(pattern))
    # 排除已转换的文件
    files = [f for f in files if "_comfyui" not in f.stem]

    if not files:
        print(f"未找到匹配的文件: {input_dir / pattern}")
        return

    print(f"找到 {len(files)} 个文件待转换\n")

    for i, input_path in enumerate(files, 1):
        output_path = output_dir / f"{input_path.stem}_comfyui.safetensors"
        print(f"[{i}/{len(files)}] {input_path.name}")
        try:
            convert_lora(str(input_path), str(output_path))
            print()
        except Exception as e:
            print(f"  转换失败: {e}\n")

    print(f"批量转换完成，共 {len(files)} 个文件")


def main():
    parser = argparse.ArgumentParser(
        description="将 Anima LoRA 转换为 ComfyUI 兼容格式"
    )
    parser.add_argument("input", help="输入的 LoRA 文件或目录")
    parser.add_argument("output", nargs="?", help="输出文件或目录 (可选)")
    parser.add_argument("--batch", "-b", action="store_true", help="批量转换目录下所有文件")
    parser.add_argument("--pattern", "-p", default="*.safetensors", help="批量模式的文件匹配模式 (默认: *.safetensors)")

    args = parser.parse_args()

    if args.batch or Path(args.input).is_dir():
        batch_convert(args.input, args.output, args.pattern)
    else:
        convert_lora(args.input, args.output)


if __name__ == "__main__":
    main()
