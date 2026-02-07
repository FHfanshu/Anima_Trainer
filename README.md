# Anima Trainer v1.02

Anima (Cosmos-Predict2) LoRA/LoKr trainer for Windows.

## Overview

- Single-file trainer entry: `anima_train.py`
- Config via TOML: `--config ./anima_lora_config.toml`
- Supports `lora` and `lokr`
- Optional ComfyUI key conversion on save
- Optional W&B logging

## Repository Layout

- `anima_train.py`: main training script
- `config_loader.py`: TOML config loader
- `anima_lora_config.example.toml`: generic config template
- `install_dependencies.bat`: one-click dependency installer (Windows)
- `convert_lora_to_comfyui.py`: LoRA key conversion utility
- `fix_comfyui_lora_keys.py`: key-fix helper
- `models/`: modeling code
- `anima/text_encoders/`: tokenizer/config assets
- `gui/`: optional GUI module

## Quick Start

1. Clone repo and enter directory.
2. Run dependency installer:

```bat
install_dependencies.bat
```

3. Create your local config from template:

```bat
copy anima_lora_config.example.toml anima_lora_config.toml
```

4. Edit `anima_lora_config.toml`:
- Set model paths in `[model]`
- Set dataset path in `[dataset].data_dir`
- Set output name/path in `[lora]` and `[output]`

5. Start training:

```bat
.venv\Scripts\python.exe anima_train.py --config .\anima_lora_config.toml
```

## Model Files

Weight files are not stored in this repository.
Place your local model files under paths you configure, for example:

- `anima/diffusion_models/*.safetensors`
- `anima/vae/*.safetensors`
- `anima/text_encoders/` (tokenizer + encoder files)

## Notes

- `anima_lora_config.toml` is intentionally ignored and should stay local.
- Use `anima_lora_config.example.toml` as the shared template.
- For `adamw8bit`, install `bitsandbytes` separately if needed.

## Contributors

- [FHfanshu](https://github.com/FHfanshu)
- OpenAI Codex

## Acknowledgements

- Community references for LoRA/LoKr training workflows
- This training script references community-circulated implementations (including upper directory v1.01). If any content infringes your rights, please contact for removal.
- PyTorch, Transformers, LyCORIS, Weights & Biases

## License

MIT License for this repository code.
Please follow original licenses/terms for upstream models and third-party assets.
