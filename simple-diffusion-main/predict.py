# predict.py (最终版 - 输出单通道灰度图)

import argparse
import torch
import yaml
from types import SimpleNamespace
import numpy as np
import tifffile
from tqdm import tqdm
from pathlib import Path

from simple_diffusion.model import UNet
from simple_diffusion.scheduler import DDIMScheduler


def preprocess_single_frame(frame_np, device):
    tensor = torch.from_numpy(frame_np.astype(np.float32) / 65535.0)
    tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


def postprocess_single_frame_grayscale(frame_tensor):
    """
    对单帧的输出张量进行后处理，取3通道平均值，并转换回16-bit NumPy图像。
    """
    # [核心修改] 在通道维度上取平均值，并移除通道维度
    frame_tensor_gray = frame_tensor.mean(dim=0, keepdim=False)

    # 将 [0, 1] 范围转换回 [0, 65535]
    frame_np_gray = (frame_tensor_gray.cpu().numpy() * 65535.0).astype(np.uint16)
    return frame_np_gray


def main(config):
    with open(config.train_config_path, 'r', encoding='utf-8') as f:
        train_config = SimpleNamespace(**yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.manual_seed(config.seed)
    print(f"使用设备: {device}")

    model = UNet(3, image_size=train_config.image_size, hidden_dims=[64, 128, 256, 512],
                 use_flash_attn=train_config.use_flash_attn).to(device)
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine")

    print(f"正在从 {config.checkpoint_path} 加载模型权重...")
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    state_dict_key = 'ema_model_state' if 'ema_model_state' in checkpoint else 'model_state'
    model.load_state_dict(checkpoint[state_dict_key])
    print(f"已成功加载 '{state_dict_key}' 的权重。")
    model.eval()

    print(f"正在加载输入文件: {config.input_tif_path}")
    input_stack = tifffile.imread(config.input_tif_path)
    if input_stack.ndim != 3:
        raise ValueError(f"输入文件必须是一个3D堆栈 (frames, height, width)，但检测到维度为 {input_stack.ndim}")

    num_frames = input_stack.shape[0]
    print(f"检测到 {num_frames} 帧图像，开始逐帧处理...")

    predicted_frames = []
    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="正在预测帧"):
            single_frame_np = input_stack[i]
            low_res_tensor = preprocess_single_frame(single_frame_np, device)

            generated_output = noise_scheduler.generate(
                model, num_inference_steps=config.inference_steps,
                generator=generator, batch_size=1, condition=low_res_tensor
            )

            predicted_frame_tensor = generated_output["sample_pt"][0]
            # [核心修改] 调用新的后处理函数
            predicted_frames.append(postprocess_single_frame_grayscale(predicted_frame_tensor))

    final_output_stack = np.stack(predicted_frames, axis=0)  # Shape: (9, 64, 64)
    print(f"所有帧处理完毕，正在保存结果到: {config.output_tif_path}")

    output_path = Path(config.output_tif_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(output_path, final_output_stack)
    print("预测完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="predict.yaml", help="Path to the prediction config file.")
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = SimpleNamespace(**yaml.safe_load(f))
    main(config)