# Anima Trainer v1.02（中文说明）

英文 README: [README.md](./README.md)

## 项目简介

Anima（Cosmos-Predict2）LoRA/LoKr 训练脚本（Windows）。

## 快速开始

1. 安装依赖：

```bat
install_dependencies.bat
```

2. 复制配置模板：

```bat
copy anima_lora_config.example.toml anima_lora_config.toml
```

3. 修改 `anima_lora_config.toml` 中模型路径、数据集路径和输出参数。

4. 启动训练：

```bat
.venv\Scripts\python.exe anima_train.py --config .\anima_lora_config.toml
```

## 说明

- 仓库不包含模型权重文件。
- 本地配置 `anima_lora_config.toml` 默认不上传。
- 训练脚本参考了网上流传实现（含上级目录 v1.01 版本）；如有侵权请联系删除。
