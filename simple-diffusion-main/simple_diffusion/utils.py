# simple_diffusion/utils.py (最终完整版)

import os
from datetime import datetime
from PIL import Image
import torch
import numpy as np
from torchvision import utils


def unnormalize_to_zero_to_one(t):
    """
    将 [-1, 1] 范围的张量转换回 [0, 1] 范围。
    这个函数被 ddim.py 依赖。
    """
    return (t + 1) * 0.5


def save_images(generated_images, epoch, args):
    """
    保存训练过程中生成的图像样本。
    'args' 是一个从 config.yaml 加载的 SimpleNamespace 对象。
    这个函数已经被修改以适应我们的项目。
    """
    # 'generate' 函数返回的 sample_pt 是 [0, 1] 范围的 PyTorch 张量
    images_tensor = generated_images["sample_pt"]

    # 'generate' 函数返回的 sample 是 [0, 1] 范围的 NumPy 数组
    images_numpy = generated_images["sample"]
    # 将 NumPy 数组转换到 [0, 255] 的整数范围以便保存为图片文件
    images_processed = (images_numpy * 255).round().astype("uint8")

    # 基于模型输出文件名创建一个清晰的目录名
    model_name_prefix = os.path.basename(args.output_dir).split('.')[0]
    out_dir = os.path.join(args.samples_dir, f"{model_name_prefix}_epoch_{epoch}")
    os.makedirs(out_dir, exist_ok=True)

    # 逐个保存单张图片
    for idx, image_np in enumerate(images_processed):
        image_pil = Image.fromarray(image_np)
        image_pil.save(os.path.join(out_dir, f"{idx:03d}.png"))

    # 将所有样本图片拼接成一张网格图，方便快速预览
    utils.save_image(images_tensor,
                     os.path.join(out_dir, "grid.png"),
                     nrow=4)  # 每行显示4张图片


# --- 以下是原始库中的其他辅助函数，保留它们以确保代码完整性 ---

def numpy_to_pil(images):
    """将 NumPy 图像数组转换为 PIL 图像列表。"""
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def match_shape(values, broadcast_array, tensor_format="pt"):
    """确保 'values' 的形状可以与 'broadcast_array' 进行广播操作。"""
    values = values.flatten()
    while len(values.shape) < len(broadcast_array.shape):
        values = values[..., None]
    if tensor_format == "pt":
        values = values.to(broadcast_array.device)
    return values


def clip(tensor, min_value=None, max_value=None):
    """根据类型（NumPy 或 PyTorch）裁剪张量的值。"""
    if isinstance(tensor, np.ndarray):
        return np.clip(tensor, min_value, max_value)
    elif isinstance(tensor, torch.Tensor):
        return torch.clamp(tensor, min_value, max_value)
    raise ValueError(f"Tensor format is not valid - should be numpy array or torch tensor. Got {type(tensor)}.")