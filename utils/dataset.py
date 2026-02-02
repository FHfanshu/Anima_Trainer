"""
Dataset Module - 标签数据集处理
===============================
处理 Anima 风格的 tag-based 数据集
每个图片对应一个同名的 .txt 文件，包含 Danbooru 风格的标签

为什么使用 Dataset 类：
1. 统一数据加载接口
2. 支持数据预处理和增强
3. 兼容 PyTorch DataLoader
4. 支持缓存和惰性加载
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class TagBasedDataset(Dataset):
    """
    Tag-based 数据集类
    
    数据集结构：
    data_root/
        ├── image1.jpg
        ├── image1.txt
        ├── image2.png
        ├── image2.txt
        └── ...
    
    .txt 文件格式（Danbooru 风格）：
    1girl, oomuro sakurako, yuru yuri, brown hair, long hair, smile, ...
    
    特性：
    - 支持多种图像格式（jpg, png, webp, jxl 等）
    - 支持 tag dropout（随机丢弃标签以增强泛化）
    - 支持图像数据增强（随机裁剪、翻转等）
    - 支持多分辨率训练
    """
    
    # 支持的图像扩展名
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.jxl'}
    
    def __init__(
        self,
        data_root: str,
        tokenizer,
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: bool = True,
        tag_dropout: float = 0.1,
        max_length: int = 77,
    ):
        """
        初始化数据集
        
        Args:
            data_root: 数据集根目录路径
            tokenizer: 文本编码器的 tokenizer
            resolution: 训练图像分辨率（正方形）
            center_crop: 是否中心裁剪图像
            random_flip: 是否随机水平翻转
            tag_dropout: 标签丢弃概率（0-1）
            max_length: 最大 token 长度
        """
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.tag_dropout = tag_dropout
        self.max_length = max_length
        
        # ------------------------------------------------------------------
        # 扫描数据集
        # ------------------------------------------------------------------
        # 找到所有有效的图像-标签对
        self.image_paths = []
        self.caption_paths = []
        
        self._scan_dataset()
        
        if len(self.image_paths) == 0:
            raise ValueError(
                f"在 {data_root} 中没有找到有效的图像-标签对。\n"
                f"请确保：\n"
                f"1. 目录中存在图片文件（支持的格式：{self.SUPPORTED_EXTENSIONS}）\n"
                f"2. 每个图片都有同名的 .txt 标签文件\n"
                f"例如：image.jpg 和 image.txt"
            )
        
        print(f"找到 {len(self.image_paths)} 个训练样本")
        
        # ------------------------------------------------------------------
        # 设置图像变换
        # ------------------------------------------------------------------
        # 变换流程：
        # 1. 调整大小（保持长宽比）
        # 2. 裁剪（中心或随机）
        # 3. 随机翻转（数据增强）
        # 4. 转换为张量
        # 5. 归一化到 [-1, 1]
        
        self.transforms = self._build_transforms()
    
    def _scan_dataset(self):
        """
        扫描数据集目录，找到所有有效的图像-标签对
        
        为什么需要扫描：
        1. 验证数据完整性
        2. 提前发现缺失的标签文件
        3. 构建索引，加速后续访问
        """
        for ext in self.SUPPORTED_EXTENSIONS:
            for image_path in self.data_root.glob(f"*{ext}"):
                # 检查对应的 .txt 文件是否存在
                caption_path = image_path.with_suffix('.txt')
                if caption_path.exists():
                    self.image_paths.append(image_path)
                    self.caption_paths.append(caption_path)
                else:
                    # 尝试其他常见命名格式
                    alt_caption_path = image_path.with_suffix('.caption')
                    if alt_caption_path.exists():
                        self.image_paths.append(image_path)
                        self.caption_paths.append(alt_caption_path)
    
    def _build_transforms(self):
        """
        构建图像变换管道
        
        返回:
            transforms.Compose: 变换管道
        """
        transform_list = []
        
        # 第一步：调整大小
        # 使用 LANCZOS 插值算法，保持高质量
        if self.center_crop:
            # 中心裁剪模式：先将短边调整为 resolution
            transform_list.append(
                transforms.Resize(
                    self.resolution,
                    interpolation=transforms.InterpolationMode.LANCZOS
                )
            )
            transform_list.append(transforms.CenterCrop(self.resolution))
        else:
            # 随机裁剪模式：先将长边调整为 resolution
            transform_list.append(
                transforms.Resize(
                    self.resolution,
                    interpolation=transforms.InterpolationMode.LANCZOS
                )
            )
            transform_list.append(transforms.RandomCrop(self.resolution))
        
        # 第二步：随机水平翻转（数据增强）
        if self.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # 第三步：转换为张量
        transform_list.append(transforms.ToTensor())
        
        # 第四步：归一化到 [-1, 1]
        # VAE 期望输入范围是 [-1, 1]，而不是 [0, 1]
        transform_list.append(transforms.Normalize([0.5], [0.5]))
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def _load_caption(self, caption_path: Path) -> str:
        """
        加载并处理标签文件
        
        处理流程：
        1. 读取文件内容
        2. 处理 tag dropout（随机丢弃部分标签）
        3. 清理格式
        
        Args:
            caption_path: 标签文件路径
            
        Returns:
            str: 处理后的标签字符串
        """
        # 读取标签
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        # 如果启用 tag dropout，随机丢弃部分标签
        if self.tag_dropout > 0 and caption:
            tags = [tag.strip() for tag in caption.split(',')]
            
            # 保留至少一个标签（避免空提示）
            if len(tags) > 1:
                # 随机丢弃标签
                kept_tags = [
                    tag for tag in tags
                    if random.random() > self.tag_dropout
                ]
                
                # 确保至少保留一个标签
                if len(kept_tags) == 0:
                    kept_tags = [random.choice(tags)]
                
                caption = ', '.join(kept_tags)
        
        return caption
    
    def _tokenize_caption(self, caption: str) -> torch.Tensor:
        """
        使用 tokenizer 编码标签
        
        Args:
            caption: 标签字符串
            
        Returns:
            torch.Tensor: token IDs
        """
        # 编码标签
        # padding="max_length" 确保所有序列长度一致
        # truncation=True 截断超长序列
        inputs = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return inputs.input_ids.squeeze(0)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            Dict: 包含以下键的字典
                - pixel_values: 图像张量 [3, H, W]
                - input_ids: token IDs [max_length]
                - caption: 原始标签字符串（用于调试）
        """
        # 加载图像
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"警告：无法加载图像 {image_path}，错误：{e}")
            # 返回一个空白图像作为占位符
            image = Image.new('RGB', (self.resolution, self.resolution), (128, 128, 128))
        
        # 应用变换
        pixel_values = self.transforms(image)
        
        # 加载标签
        caption_path = self.caption_paths[idx]
        caption = self._load_caption(caption_path)
        
        # 编码标签
        input_ids = self._tokenize_caption(caption)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "caption": caption,  # 用于调试和日志
        }


class CachedTagBasedDataset(TagBasedDataset):
    """
    带缓存的 TagBasedDataset
    
    特性：
    - 缓存预处理后的 latent（节省 VAE 编码时间）
    - 缓存 tokenized caption
    - 首次加载较慢，后续加载快
    
    适用场景：
    - 数据集不太大（可以放入内存）
    - 需要多次 epoch 训练
    - 想要最大化 GPU 利用率
    """
    
    def __init__(
        self,
        data_root: str,
        tokenizer,
        vae,
        device,
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: bool = True,
        tag_dropout: float = 0.1,
        max_length: int = 77,
        cache_dir: Optional[str] = None,
    ):
        """
        初始化缓存数据集
        
        Args:
            vae: VAE 模型，用于预编码图像到 latent
            device: 计算设备
            cache_dir: 缓存目录，如果为 None 则使用内存缓存
        """
        super().__init__(
            data_root=data_root,
            tokenizer=tokenizer,
            resolution=resolution,
            center_crop=center_crop,
            random_flip=random_flip,
            tag_dropout=tag_dropout,
            max_length=max_length,
        )
        
        self.vae = vae
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # 内存缓存
        self.latents_cache = {}
        self.tokens_cache = {}
        
        # 如果使用磁盘缓存，创建目录
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()
        
        # 预计算所有样本
        self._precompute_cache()
    
    def _load_disk_cache(self):
        """从磁盘加载缓存（如果存在）"""
        # TODO: 实现磁盘缓存加载
        pass
    
    def _precompute_cache(self):
        """
        预计算所有样本的 latent 和 tokens
        
        警告：这会消耗大量内存，确保数据集不会太大
        """
        print("正在预计算缓存...")
        
        self.vae.eval()
        with torch.no_grad():
            for idx in range(len(self)):
                # 获取原始数据
                sample = super().__getitem__(idx)
                
                # 编码图像到 latent
                pixel_values = sample["pixel_values"].unsqueeze(0).to(self.device)
                latent = self.vae.encode(pixel_values).latent_dist.sample()
                latent = latent * self.vae.config.scaling_factor
                latent = latent.squeeze(0).cpu()
                
                # 缓存
                self.latents_cache[idx] = latent
                self.tokens_cache[idx] = sample["input_ids"]
        
        print(f"缓存完成！占用内存: ~{self._estimate_cache_size():.2f} GB")
    
    def _estimate_cache_size(self) -> float:
        """估计缓存占用的内存大小（GB）"""
        if len(self.latents_cache) == 0:
            return 0.0
        
        # 计算一个 latent 的大小
        sample_latent = next(iter(self.latents_cache.values()))
        latent_size = sample_latent.numel() * sample_latent.element_size()
        
        # 计算总大小
        total_bytes = latent_size * len(self.latents_cache)
        
        # 加上 tokens 的大小（相对很小）
        sample_tokens = next(iter(self.tokens_cache.values()))
        tokens_size = sample_tokens.numel() * sample_tokens.element_size()
        total_bytes += tokens_size * len(self.tokens_cache)
        
        return total_bytes / (1024 ** 3)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取缓存的数据"""
        # 应用随机翻转（必须在缓存后应用，因为翻转是随机的）
        latent = self.latents_cache[idx].clone()
        
        if self.random_flip and random.random() < 0.5:
            # 水平翻转 latent
            latent = torch.flip(latent, dims=[-1])
        
        # 应用 tag dropout
        caption_path = self.caption_paths[idx]
        caption = self._load_caption(caption_path)
        input_ids = self._tokenize_caption(caption)
        
        return {
            "latents": latent,
            "input_ids": input_ids,
            "caption": caption,
        }


def collate_fn(examples):
    """
    DataLoader 的 collate 函数
    
    将一批样本组合成 batch tensor
    
    Args:
        examples: 样本列表，每个样本是 __getitem__ 返回的字典
        
    Returns:
        Dict: batch 数据
    """
    # 堆叠图像
    if "pixel_values" in examples[0]:
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        pixel_values = None
    
    # 堆叠 latent（如果使用缓存数据集）
    if "latents" in examples[0]:
        latents = torch.stack([example["latents"] for example in examples])
        latents = latents.to(memory_format=torch.contiguous_format).float()
    else:
        latents = None
    
    # 堆叠 token IDs
    input_ids = torch.stack([example["input_ids"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "latents": latents,
        "input_ids": input_ids,
    }
