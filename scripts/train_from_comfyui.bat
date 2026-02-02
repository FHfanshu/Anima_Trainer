@echo off
REM 从 ComfyUI 目录训练 Anima LoRA
REM ==============================

REM 设置 ComfyUI 路径
set COMFYUI_PATH=D:\ComfyUI

REM 检查路径是否存在
if not exist "%COMFYUI_PATH%" (
    echo Error: ComfyUI path not found: %COMFYUI_PATH%
    echo Please edit this script and set the correct path.
    pause
    exit /b 1
)

echo Using ComfyUI path: %COMFYUI_PATH%
echo.

REM 启动训练
accelerate launch train.py ^
  --comfyui_path="%COMFYUI_PATH%" ^
  --data_root="./data/character_dataset" ^
  --output_dir="./output/character_lora_comfyui" ^
  --lora_rank=32 ^
  --lora_alpha=32 ^
  --num_train_epochs=100 ^
  --learning_rate=1e-4 ^
  --gradient_checkpointing ^
  --enable_flash_attention ^
  --optimizer=adamw8bit ^
  --mixed_precision=bf16

echo.
echo Training completed!
pause