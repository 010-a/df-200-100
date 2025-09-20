import torch
from torch.utils.data import Dataset
from PIL import Image
import tifffile as tiff
from pathlib import Path
import numpy as np


class FluorescenceTIFDataset(Dataset):
    def __init__(self, data_root, hr_size, lr_size, transform=None):
        """
        Args:
            data_root (str): 数据集根目录 (例如 'data/train')
            hr_size (int): 高分辨率图像的目标尺寸
            lr_size (int): 低分辨率图像的目标尺寸
        """
        self.data_root = Path(data_root)
        self.hr_dir = self.data_root / "hr"
        self.lr_dir = self.data_root / "lr"

        # 获取所有图像路径并排序，确保lr和hr一一对应
        self.lr_paths = sorted([p for p in self.lr_dir.glob("*.tif")])
        self.hr_paths = sorted([p for p in self.hr_dir.glob("*.tif")])

        # 确保文件数量匹配
        assert len(self.lr_paths) == len(self.hr_paths), "Low-res and High-res image counts do not match!"
        assert len(self.lr_paths) > 0, f"No images found in {self.lr_dir}"

        self.hr_size = hr_size
        self.lr_size = lr_size

        # 注意：这里的transform可以留作之后的数据增强，我们直接在代码中处理核心转换
        self.transform = transform

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_path = self.lr_paths[idx]
        hr_path = self.hr_paths[idx]

        # 使用tifffile读取16-bit TIF图像
        lr_image = tiff.imread(lr_path)
        hr_image = tiff.imread(hr_path)

        # --- 核心转换逻辑 ---
        # 1. 转换为浮点数并归一化到 [0, 1]
        # 16-bit的最大值是 2^16 - 1 = 65535
        lr_image = lr_image.astype(np.float32) / 65535.0
        hr_image = hr_image.astype(np.float32) / 65535.0

        # 2. 转换为PyTorch Tensor
        # (H, W) -> (1, H, W)
        lr_tensor = torch.from_numpy(lr_image).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_image).unsqueeze(0)

        # 3. 如果是单通道灰度图，复制成3通道，以匹配U-Net的输入
        # 这是一个常见的做法，因为预训练模型通常基于RGB图像
        if lr_tensor.shape[0] == 1:
            lr_tensor = lr_tensor.repeat(3, 1, 1)
        if hr_tensor.shape[0] == 1:
            hr_tensor = hr_tensor.repeat(3, 1, 1)

        # 4. 调整尺寸 (如果需要)
        # 您的数据已经是64x64和128x128，所以这里可以省略，但代码保留了这种能力
        # lr_tensor = F.interpolate(lr_tensor.unsqueeze(0), size=(self.lr_size, self.lr_size), mode='bilinear', align_corners=False).squeeze(0)
        # hr_tensor = F.interpolate(hr_tensor.unsqueeze(0), size=(self.hr_size, self.hr_size), mode='bilinear', align_corners=False).squeeze(0)

        # 5. 归一化到 [-1, 1]，这是扩散模型的标准做法
        lr_tensor = lr_tensor * 2.0 - 1.0
        hr_tensor = hr_tensor * 2.0 - 1.0

        return {"low_res": lr_tensor, "high_res": hr_tensor}