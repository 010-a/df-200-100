# simple_diffusion/utils.py (修复后版本)

import os
from datetime import datetime
from PIL import Image
import torch
import numpy as np
from torchvision import utils


def save_images(generated_images, epoch, args):
    """
    保存生成的图像样本。
    'args' 现在是一个 SimpleNamespace 对象，从 config.yaml 加载。
    """
    images = generated_images["sample"]
    # 将图像从 [0, 1] 范围转换回 [0, 255] 整数范围
    images_processed = (images * 255).round().astype("uint8")

    current_date = datetime.today().strftime('%Y%m%d_%H%M%S')

    # --- [核心修复] ---
    # 不再使用 'dataset_name'，因为我们的config里没有这个字段。
    # 创建一个更通用的文件夹名，例如 'epoch_0_step_1000'
    # 'os.path.basename(args.output_dir).split('.')[0]' 会从模型路径中提取文件名，例如 'fluorescence_superres'
    model_name_prefix = os.path.basename(args.output_dir).split('.')[0]
    out_dir = os.path.join(args.samples_dir, f"{model_name_prefix}_epoch_{epoch}")
    # --- 修复结束 ---

    os.makedirs(out_dir, exist_ok=True)

    for idx, image in enumerate(images_processed):
        image = Image.fromarray(image)
        # 保存为PNG格式，因为它无损且通用
        image.save(os.path.join(out_dir, f"{idx:03d}.png"))

    # 将所有样本图片拼接成一张网格图，方便预览
    utils.save_image(generated_images["sample_pt"],
                     os.path.join(out_dir, "grid.png"),
                     nrow=4)  # 每行显示4张图片

# (删除了原文件中其他不再使用的函数)