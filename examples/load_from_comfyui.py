#!/usr/bin/env python3
"""
ComfyUI 加载示例
==============
演示如何从 ComfyUI 目录加载 Anima 模型进行训练
"""

from utils.comfyui_loader import from_comfyui, find_comfyui_models

# 方法 1: 简单加载
comfyui_path = "D:/ComfyUI"  # 修改为你的 ComfyUI 路径

print("=" * 60)
print("方法 1: 直接从 ComfyUI 加载")
print("=" * 60)

pipeline = from_comfyui(
    comfyui_path=comfyui_path,
    torch_dtype="bfloat16",  # RTX 3090 推荐
    device="cuda",
)

print("\n✓ 模型加载成功！")
print(f"  Transformer: {pipeline.transformer.__class__.__name__}")
print(f"  VAE: {pipeline.vae.__class__.__name__}")
print(f"  Device: {pipeline.device}")

# 方法 2: 先查找模型再加载
print("\n" + "=" * 60)
print("方法 2: 查找模型文件")
print("=" * 60)

models = find_comfyui_models(comfyui_path)
for key, path in models.items():
    print(f"  {key}: {path}")

# 方法 3: 使用智能加载（自动尝试多个来源）
print("\n" + "=" * 60)
print("方法 3: 智能加载（自动回退）")
print("=" * 60)

from utils.comfyui_loader import load_anima_with_fallback

pipeline = load_anima_with_fallback(
    comfyui_path=comfyui_path,  # 优先尝试
    # pretrained_model_name_or_path="circlestone-labs/Anima",  # 后备选项
)

print("\n✓ 智能加载成功！")

# 测试推理
print("\n" + "=" * 60)
print("测试推理")
print("=" * 60)

prompt = "1girl, oomuro sakurako, yuru yuri, smile, brown hair, best quality"
print(f"Prompt: {prompt}")

with torch.no_grad():
    image = pipeline(
        prompt=prompt,
        num_inference_steps=30,
        height=1024,
        width=1024,
    ).images[0]

image.save("test_output.png")
print("✓ 测试图像已保存到 test_output.png")
