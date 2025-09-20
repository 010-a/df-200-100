# simple_diffusion/dataset.py

import torch
from torch.utils.data import Dataset
import tifffile as tiff
from pathlib import Path
import numpy as np
import torch.nn.functional as F


class MultiFrameTIFDataset(Dataset):
    """
    一个专门用于处理多帧16-bit TIF荧光图像数据的数据集类。
    假设每个TIF文件都包含相同数量的帧（例如9帧）。
    """

    def __init__(self, data_root, frames_per_file=9):
        """
        Args:
            data_root (str): 数据集根目录 (例如 'data/train')。
                             该目录下应包含 'lr' 和 'hr' 两个子文件夹。
            frames_per_file (int): 每个TIF文件包含的帧数。
        """
        self.data_root = Path(data_root)
        self.lr_dir = self.data_root / "lr"
        self.hr_dir = self.data_root / "hr"
        self.frames_per_file = frames_per_file

        # 获取所有图像路径并排序，确保lr和hr文件一一对应
        self.lr_paths = sorted([p for p in self.lr_dir.glob("*.tif")])
        self.hr_paths = sorted([p for p in self.hr_dir.glob("*.tif")])

        # 断言检查，确保数据完整性
        assert len(self.lr_paths) == len(self.hr_paths), \
            f"低分辨率和高分辨率图像文件数量不匹配于: {data_root}"
        assert len(self.lr_paths) > 0, f"在 {self.lr_dir} 中未找到TIF文件"

        # 数据集的总长度是 文件数 * 每文件的帧数
        self.total_samples = len(self.lr_paths) * self.frames_per_file

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        根据给定的索引，加载对应的低分辨率和高分辨率图像帧。
        """
        # --- 1. 计算文件索引和帧索引 ---
        file_idx = idx // self.frames_per_file
        frame_idx = idx % self.frames_per_file

        lr_path = self.lr_paths[file_idx]
        hr_path = self.hr_paths[file_idx]

        # --- 2. 使用tifffile读取整个TIF堆栈 ---
        # 这会返回一个 (frames, height, width) 的NumPy数组
        lr_stack = tiff.imread(lr_path)
        hr_stack = tiff.imread(hr_path)

        # 提取对应的帧
        lr_image = lr_stack[frame_idx]
        hr_image = hr_stack[frame_idx]

        # --- 3. 图像预处理 ---
        # a. 转换为32位浮点数，并归一化到 [0, 1]
        # 16-bit TIF 的最大值是 2^16 - 1 = 65535
        lr_image = lr_image.astype(np.float32) / 65535.0
        hr_image = hr_image.astype(np.float32) / 65535.0

        # b. 转换为PyTorch Tensor并增加通道维度 (H, W) -> (1, H, W)
        lr_tensor = torch.from_numpy(lr_image).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_image).unsqueeze(0)

        # c. 从单通道复制到三通道，以匹配U-Net的输入要求
        lr_tensor = lr_tensor.repeat(3, 1, 1)
        hr_tensor = hr_tensor.repeat(3, 1, 1)

        # d. 归一化到 [-1, 1]，这是扩散模型的标准输入范围
        lr_tensor = lr_tensor * 2.0 - 1.0
        hr_tensor = hr_tensor * 2.0 - 1.0

        return {"low_res": lr_tensor, "high_res": hr_tensor}

# 注意：删除了原文件中所有其他的Dataset类，只保留我们需要的这一个。