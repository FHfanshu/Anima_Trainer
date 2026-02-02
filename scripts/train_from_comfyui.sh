#!/bin/bash
# 从 ComfyUI 目录训练 Anima LoRA
# ==============================

# 设置 ComfyUI 路径
COMFYUI_PATH="${COMFYUI_PATH:-$HOME/ComfyUI}"

# 检查路径是否存在
if [ ! -d "$COMFYUI_PATH" ]; then
    echo "Error: ComfyUI path not found: $COMFYUI_PATH"
    echo "Please set COMFYUI_PATH environment variable or edit this script."
    echo ""
    echo "Example:"
    echo "  export COMFYUI_PATH=/path/to/ComfyUI"
    echo "  bash $0"
    exit 1
fi

echo "Using ComfyUI path: $COMFYUI_PATH"
echo ""

# 启动训练
accelerate launch train.py \
  --comfyui_path="$COMFYUI_PATH" \
  --data_root="./data/character_dataset" \
  --output_dir="./output/character_lora_comfyui" \
  --lora_rank=32 \
  --lora_alpha=32 \
  --num_train_epochs=100 \
  --learning_rate=1e-4 \
  --gradient_checkpointing \
  --enable_flash_attention \
  --optimizer=adamw8bit \
  --mixed_precision=bf16

echo ""
echo "Training completed!"