# Anima Character LoRA Training Script

一个完整的、生产级的 **Anima (Cosmos 架构)** Character LoRA 训练脚本，支持标准 PEFT LoRA 和 LyCORIS Lokr 两种模式。

## 特性

- **双模式 LoRA 支持**: 标准 PEFT LoRA 和 LyCORIS Lokr
- **双模式优化器**: 8-bit AdamW (bitsandbytes) 和 muon 优化器
- **内存优化**: Gradient Checkpointing + Flash Attention 2
- **标签数据集**: 支持 Danbooru 风格的 tag-based 数据集
- **完整训练管理**: WandB 日志、自动 checkpoint、resume 支持
- **单卡优化**: 专为 RTX 3090 优化，可训练 1024x1024 分辨率

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd Anima_Trainer

# 创建虚拟环境（推荐）
conda create -n anima-lora python=3.10
conda activate anima-lora

# 安装依赖
pip install -r requirements.txt

# 安装 Flash Attention 2（可选，但强烈推荐）
pip install flash-attn --no-build-isolation
```

## 快速开始

### 1. 准备数据集

将数据集组织为以下结构：

```
data/
└── character_dataset/
    ├── image001.jpg
    ├── image001.txt
    ├── image002.png
    ├── image002.txt
    └── ...
```

`.txt` 文件包含 Danbooru 风格的标签，例如：

```
1girl, oomuro sakurako, yuru yuri, brown hair, long hair, smile, school uniform, ...
```

### 2. 配置训练参数

编辑 `config/train_config.yaml` 或直接修改启动脚本。

### 3. 启动训练

**Linux/macOS:**
```bash
bash scripts/train_example.sh
```

**Windows:**
```batch
scripts\train_example.bat
```

**直接使用 accelerate:**
```bash
accelerate launch train.py \
  --pretrained_model_name_or_path="circlestone-labs/Anima" \
  --data_root="./data/character_dataset" \
  --output_dir="./output/character_lora" \
  --lora_rank=32 \
  --lora_alpha=32 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=100 \
  --learning_rate=1e-4 \
  --gradient_checkpointing \
  --enable_flash_attention \
  --checkpointing_steps=500 \
  --mixed_precision=bf16 \
  --optimizer=adamw8bit
```

## 项目结构

```
Anima_Trainer/
├── train.py                  # 主训练脚本
├── requirements.txt          # 依赖列表
├── config/
│   └── train_config.yaml    # 训练配置示例
├── utils/
│   ├── __init__.py
│   ├── dataset.py           # 数据集处理
│   ├── model_utils.py       # 模型加载和 LoRA 配置
│   ├── optimizer_utils.py   # 优化器创建
│   └── checkpoint.py        # Checkpoint 管理
└── scripts/
    ├── train_example.sh     # Linux/macOS 启动脚本
    └── train_example.bat    # Windows 启动脚本
```

## 配置详解

### LoRA 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lora_type` | LoRA 类型: `lora` 或 `lokr` | `lora` |
| `--lora_rank` | LoRA rank (4-128) | `32` |
| `--lora_alpha` | LoRA alpha (通常为 rank 的 1-2 倍) | `32` |
| `--lora_dropout` | Dropout 概率 | `0.0` |

### 优化器选择

| 优化器 | 命令 | 显存节省 | 适用场景 |
|--------|------|----------|----------|
| 8-bit AdamW | `--optimizer=adamw8bit` | ~50% | 显存受限 |
| Muon | `--optimizer=muon` | 标准 | 实验性 |
| 标准 AdamW | `--optimizer=adamw` | 标准 | 后备选项 |

### 内存优化

- **Gradient Checkpointing**: `--gradient_checkpointing` (推荐开启)
- **Flash Attention 2**: `--enable_flash_attention` (需要 `flash-attn` 库)
- **8-bit Optimizer**: `--optimizer=adamw8bit`

### Resume from Checkpoint

```bash
accelerate launch train.py \
  --resume_from_checkpoint="./output/checkpoints/checkpoint-1000" \
  ...
```

## 硬件要求

- **GPU**: RTX 3090 (24GB) 或更高
- **内存**: 32GB+ RAM
- **存储**: 至少 50GB 可用空间

## 最佳实践

### 1. 学习率

- 单卡 RTX 3090: `1e-4` ~ `5e-4`
- 使用梯度累积时: `1e-4` (已自动缩放)

### 2. LoRA Rank

- 角色训练: `rank=32` ~ `64`, `alpha=32` ~ `64`
- 风格训练: `rank=64` ~ `128`, `alpha=64` ~ `128`

### 3. 数据集大小

- 最小推荐: 20-30 张高质量图片
- 理想范围: 50-200 张
- 标签质量比数量更重要

### 4. Tag 格式

遵循 Danbooru 格式：

```
[质量标签] [角色标签] [系列标签] [一般标签]

例如：
masterpiece, best quality, 1girl, oomuro sakurako, yuru yuri, 
brown hair, long hair, smile, school uniform, sitting, ...
```

## Troubleshooting

### 1. CUDA Out of Memory

- 降低分辨率: `--resolution=896`
- 增加梯度累积: `--gradient_accumulation_steps=8`
- 使用 8-bit 优化器: `--optimizer=adamw8bit`
- 确保启用 gradient checkpointing: `--gradient_checkpointing`

### 2. 训练不稳定

- 降低学习率: `--learning_rate=5e-5`
- 增加 warmup: `--lr_warmup_steps=1000`
- 使用 bf16 而不是 fp16: `--mixed_precision=bf16`

### 3. Checkpoint 加载失败

- 检查路径是否正确
- 确保使用相同的模型和 LoRA 配置
- 查看日志获取详细错误信息

## 高级用法

### 使用配置文件

```bash
# 使用 OmegaConf 加载 YAML 配置
accelerate launch train.py --config_file=config/train_config.yaml
```

### 多卡训练

```bash
accelerate launch --multi_gpu --num_processes=2 train.py ...
```

### 自定义验证提示词

```bash
--validation_prompt="1girl, sakurako, masterpiece, best quality" \
--validation_epochs=5 \
--num_validation_images=4
```

## 训练后使用 LoRA

生成的 LoRA 文件可以使用以下方式加载：

### ComfyUI
将 `lora_weights.safetensors` 放入 `ComfyUI/models/loras/`

### diffusers
```python
from diffusers import CosmosPipeline
import torch

pipe = CosmosPipeline.from_pretrained(
    "circlestone-labs/Anima",
    torch_dtype=torch.bfloat16
).to("cuda")

pipe.load_lora_weights("path/to/lora_weights.safetensors")

image = pipe(
    prompt="1girl, sakurako, masterpiece, best quality",
    num_inference_steps=30,
).images[0]
```

## License

本项目采用 MIT License。Anima 模型本身的使用请遵循其官方许可证。

## 致谢

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

## 联系方式

如有问题或建议，欢迎提交 Issue 或 PR！
