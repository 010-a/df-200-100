# generate.py (最终版本，使用YAML配置)

import argparse
import torch
import yaml
from types import SimpleNamespace
import numpy as np
import tifffile
from PIL import Image

# 确保可以从项目根目录正确导入我们自己的模块
from simple_diffusion.model import UNet
from simple_diffusion.scheduler import DDIMScheduler


def load_and_preprocess_input(tif_path, device):
    """
    加载并预处理单张低分辨率TIF图像，使其符合模型输入要求。
    这个过程必须与 dataset.py 中的预处理完全一致。
    """
    stack = tifffile.imread(tif_path)
    image = stack[0] if stack.ndim == 3 else stack
    image = image.astype(np.float32) / 65535.0
    tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


def main(gen_config):
    # --- 1. 加载训练配置，以获取模型结构参数 ---
    with open(gen_config.train_config_path, 'r', encoding='utf-8') as f:
        train_config_dict = yaml.safe_load(f)
    train_config = SimpleNamespace(**train_config_dict)

    # --- 2. 设置环境 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.manual_seed(gen_config.seed)
    print(f"使用设备: {device}")

    # --- 3. 初始化模型和调度器 ---
    model = UNet(3, image_size=train_config.image_size, hidden_dims=[64, 128, 256, 512],
                 use_flash_attn=train_config.use_flash_attn)
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine")

    # --- 4. 加载训练好的权重 ---
    print(f"正在从 {gen_config.checkpoint_path} 加载模型权重...")
    checkpoint = torch.load(gen_config.checkpoint_path, map_location=device)

    if 'ema_model_state' in checkpoint:
        model.load_state_dict(checkpoint['ema_model_state'])
        print("已成功加载 EMA 模型权重。")
    else:
        model.load_state_dict(checkpoint['model_state'])
        print("警告: 未找到 EMA 权重，已加载标准模型权重。")

    model.to(device)
    model.eval()

    # --- 5. 加载并预处理输入图像 ---
    print(f"正在加载并预处理输入图像: {gen_config.input_tif_path}")
    low_res_tensor = load_and_preprocess_input(gen_config.input_tif_path, device)

    # --- 6. 执行推理 ---
    print(f"开始执行扩散模型反向去噪过程 (共 {gen_config.inference_steps} 步)...")
    with torch.no_grad():
        generated_images = noise_scheduler.generate(
            model,
            num_inference_steps=gen_config.inference_steps,
            generator=generator,
            eta=1.0,
            batch_size=1,
            condition=low_res_tensor
        )

    # --- 7. 后处理并保存结果 ---
    output_image_numpy = generated_images["sample"][0]
    output_image_16bit = (output_image_numpy * 65535.0).astype(np.uint16)

    tifffile.imwrite(gen_config.output_tif_path, output_image_16bit)
    print(f"推理完成！结果已保存至: {gen_config.output_tif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="generate_config.yaml",
                        help="Path to the generation config file.")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        gen_config_dict = yaml.safe_load(f)

    gen_config = SimpleNamespace(**gen_config_dict)

    main(gen_config)