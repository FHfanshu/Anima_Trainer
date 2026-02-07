@echo off
setlocal enabledelayedexpansion

rem One-click dependency installer for Anima Trainer v1.02 (Windows)

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python not found in PATH.
  echo Please install Python 3.10+ and try again.
  exit /b 1
)

set "VENV_DIR=.venv"
if not exist "%VENV_DIR%\\Scripts\\python.exe" (
  echo Creating virtual environment: %VENV_DIR%
  python -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    exit /b 1
  )
)

call "%VENV_DIR%\\Scripts\\activate.bat"
if errorlevel 1 (
  echo ERROR: Failed to activate virtual environment.
  exit /b 1
)

python -m pip install --upgrade pip

echo.
echo Select PyTorch build (stable):
echo 1. CUDA 12.1 (Recommended)
echo 2. CUDA 11.8
echo 3. CPU only
choice /c 123 /n /m "Choose 1-3: "

set "TORCH_INDEX="
if errorlevel 3 set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
if errorlevel 2 set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"
if errorlevel 1 set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"

echo.
echo Installing PyTorch and TorchVision from: %TORCH_INDEX%
python -m pip install torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 (
  echo ERROR: PyTorch install failed.
  exit /b 1
)

echo.
echo Install xformers? (stable)
echo 1. Yes
echo 2. No
choice /c 12 /n /m "Choose 1-2: "
if errorlevel 1 (
  echo.
  echo Installing xformers from: %TORCH_INDEX%
  python -m pip install xformers --index-url %TORCH_INDEX%
  if errorlevel 1 (
    echo ERROR: xformers install failed.
    exit /b 1
  )
)

echo.
echo Installing core dependencies...
python -m pip install numpy Pillow safetensors transformers einops
if errorlevel 1 (
  echo ERROR: Core dependency install failed.
  exit /b 1
)

echo.
echo Install wandb? (optional, for training logs)
echo 1. Yes
echo 2. No
choice /c 12 /n /m "Choose 1-2: "
if errorlevel 1 (
  echo.
  echo Installing wandb...
  python -m pip install wandb
  if errorlevel 1 (
    echo ERROR: wandb install failed.
    exit /b 1
  )
)

echo.
echo Install bitsandbytes? (optional, for AdamW8bit optimizer)
echo NOTE: On Windows this may require a compatible prebuilt wheel.
echo 1. Yes
echo 2. No
choice /c 12 /n /m "Choose 1-2: "
if errorlevel 1 (
  echo.
  echo Installing bitsandbytes...
  python -m pip install bitsandbytes
  if errorlevel 1 (
    echo ERROR: bitsandbytes install failed.
    exit /b 1
  )
)

echo.
echo Install prodigyopt? (optional, for Prodigy optimizer)
echo 1. Yes
echo 2. No
choice /c 12 /n /m "Choose 1-2: "
if errorlevel 1 (
  echo.
  echo Installing prodigyopt...
  python -m pip install prodigyopt
  if errorlevel 1 (
    echo ERROR: prodigyopt install failed.
    exit /b 1
  )
)

echo.
echo Done. To run training:
echo   .venv\\Scripts\\python.exe anima_train.py --config .\\anima_lora_config.toml
exit /b 0
