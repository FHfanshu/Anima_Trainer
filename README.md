# Anima LoRA Trainer

一个轻量级的 **Anima** LoRA/LoKr 训练脚本，支持 YAML 配置文件，输出兼容 ComfyUI。

## 特性

- **YAML 配置文件** - 通过 `--config` 加载配置
- **LoRA / LoKr 双模式** - 标准 LoRA 和 LyCORIS LoKr
- **ComfyUI 兼容** - 输出的 safetensors 可直接在 ComfyUI 中使用
- **VAE Latent 缓存** - 自动缓存到 npz 文件，加速后续训练
- **xformers 支持** - 可选启用 memory efficient attention
- **Rich 进度条** - 实时显示 loss 曲线

## 安装

```bash
git clone https://github.com/FHfanshu/Anima_Trainer.git
cd Anima_Trainer

# 创建虚拟环境
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install numpy Pillow safetensors transformers einops pyyaml rich tiktoken
pip install xformers --index-url https://download.pytorch.org/whl/cu130
```

## 快速开始

### 1. 准备模型文件

```
models/
├── transformers/
│   └── anima-preview.safetensors
├── vae/
│   └── qwen_image_vae.safetensors
└── text_encoders/
    ├── config.json              # 已包含在仓库中
    ├── tokenizer_config.json    # 已包含在仓库中
    └── model.safetensors        # 需下载 Qwen3-0.6B 权重
```

**注意**: `text_encoders/` 目录需要 Qwen3 模型权重文件 `model.safetensors`，可从 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main) 下载。

### 2. 准备数据集

```
dataset/
├── image001.jpg
├── image001.txt
├── image002.png
├── image002.txt
└── ...
```

`.txt` 文件包含 Danbooru 风格标签。

### 3. 编辑配置文件

复制并编辑 `config/train_template.yaml`。

### 4. 开始训练

```bash
python anima_train.py --config ./config/train_template.yaml
```

命令行参数可覆盖配置文件：

```bash
python anima_train.py --config ./config/train_template.yaml --lr 5e-5
```

## 配置说明

```yaml
# 模型路径
transformer_path: "models/transformers/anima-preview.safetensors"
vae_path: "models/vae/qwen_image_vae.safetensors"
text_encoder_path: "models/text_encoders"

# 数据集
data_dir: "./dataset"
resolution: 1024
repeats: 10
shuffle_caption: true
cache_latents: true

# LoRA
lora_type: "lokr"      # lora 或 lokr
lora_rank: 32
lora_alpha: 32

# 训练
epochs: 50
batch_size: 1
grad_accum: 4
learning_rate: 1e-4
mixed_precision: "bf16"
grad_checkpoint: true
xformers: true

# 保存
output_dir: "./output"
save_every: 5          # 每 N 个 epoch 保存
sample_every: 5        # 每 N 个 epoch 采样
```

## 硬件要求

- GPU: 24GB+ 显存 (RTX 3090/4090)
- RAM: 32GB+

## 致谢

- 本项目参考了网络上流传的 Anima 训练脚本，原作者不详，如有侵权请联系删除
- [Claude Opus 4.5](https://claude.ai) - AI 编程助手

## License

MIT License
