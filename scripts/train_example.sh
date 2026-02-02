#!/bin/bash
# Anima Character LoRA 训练脚本 - 示例
# ==================================

# 设置环境变量
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0  # 使用第一张 GPU

# 训练参数配置
MODEL_NAME="circlestone-labs/Anima"
DATA_ROOT="./data/character_dataset"
OUTPUT_DIR="./output/character_lora_r32"

# LoRA 参数
LORA_TYPE="lora"  # 或 "lokr"
LORA_RANK=32
LORA_ALPHA=32

# 训练参数
BATCH_SIZE=1
GRAD_ACCUMULATION=4
NUM_EPOCHS=100
LEARNING_RATE=1e-4
RESOLUTION=1024

# 验证参数
VALIDATION_PROMPT="1girl, oomuro sakurako, yuru yuri, brown hair, long hair, smile, best quality, masterpiece"
VALIDATION_EPOCHS=5

# WandB 参数
WANDB_PROJECT="anima-lora-training"
WANDB_NAME="sakurako_r32_test"

# 启动训练
echo "Starting Anima LoRA Training..."
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "LoRA Rank: $LORA_RANK"

accelerate launch train.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --data_root="$DATA_ROOT" \
  --output_dir="$OUTPUT_DIR" \
  --lora_type="$LORA_TYPE" \
  --lora_rank=$LORA_RANK \
  --lora_alpha=$LORA_ALPHA \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GRAD_ACCUMULATION \
  --num_train_epochs=$NUM_EPOCHS \
  --learning_rate=$LEARNING_RATE \
  --resolution=$RESOLUTION \
  --validation_prompt="$VALIDATION_PROMPT" \
  --validation_epochs=$VALIDATION_EPOCHS \
  --wandb_project="$WANDB_PROJECT" \
  --tracker_run_name="$WANDB_NAME" \
  --gradient_checkpointing \
  --enable_flash_attention \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=5 \
  --mixed_precision="bf16" \
  --optimizer="adamw8bit" \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=500 \
  --save_state

echo "Training completed!"