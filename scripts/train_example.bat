@echo off
REM Anima Character LoRA 训练脚本 - Windows 示例
REM =========================================

REM 设置环境变量
set PYTHONUNBUFFERED=1
set CUDA_VISIBLE_DEVICES=0

REM 训练参数配置
set MODEL_NAME=circlestone-labs/Anima
set DATA_ROOT=.\data\character_dataset
set OUTPUT_DIR=.\output\character_lora_r32

REM LoRA 参数
set LORA_TYPE=lora
set LORA_RANK=32
set LORA_ALPHA=32

REM 训练参数
set BATCH_SIZE=1
set GRAD_ACCUMULATION=4
set NUM_EPOCHS=100
set LEARNING_RATE=1e-4
set RESOLUTION=1024

REM 验证参数
set VALIDATION_PROMPT=1girl, oomuro sakurako, yuru yuri, brown hair, long hair, smile, best quality, masterpiece
set VALIDATION_EPOCHS=5

REM WandB 参数
set WANDB_PROJECT=anima-lora-training
set WANDB_NAME=sakurako_r32_test

echo Starting Anima LoRA Training...
echo Model: %MODEL_NAME%
echo Output: %OUTPUT_DIR%
echo LoRA Rank: %LORA_RANK%

accelerate launch train.py ^
  --pretrained_model_name_or_path="%MODEL_NAME%" ^
  --data_root="%DATA_ROOT%" ^
  --output_dir="%OUTPUT_DIR%" ^
  --lora_type="%LORA_TYPE%" ^
  --lora_rank=%LORA_RANK% ^
  --lora_alpha=%LORA_ALPHA% ^
  --train_batch_size=%BATCH_SIZE% ^
  --gradient_accumulation_steps=%GRAD_ACCUMULATION% ^
  --num_train_epochs=%NUM_EPOCHS% ^
  --learning_rate=%LEARNING_RATE% ^
  --resolution=%RESOLUTION% ^
  --validation_prompt="%VALIDATION_PROMPT%" ^
  --validation_epochs=%VALIDATION_EPOCHS% ^
  --wandb_project="%WANDB_PROJECT%" ^
  --tracker_run_name="%WANDB_NAME%" ^
  --gradient_checkpointing ^
  --enable_flash_attention ^
  --checkpointing_steps=500 ^
  --checkpoints_total_limit=5 ^
  --mixed_precision=bf16 ^
  --optimizer=adamw8bit ^
  --lr_scheduler=cosine_with_restarts ^
  --lr_warmup_steps=500 ^
  --save_state

echo Training completed!
pause