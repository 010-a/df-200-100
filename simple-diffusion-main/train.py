# train.py (最终完整版 - 以步数为中心 & 预计剩余时间)

import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import yaml
from types import SimpleNamespace
import warnings
import time
from itertools import cycle  # <-- 1. 新增导入

from diffusers.optimization import get_scheduler
from torchinfo import summary
from simple_diffusion.scheduler import DDIMScheduler
from simple_diffusion.model import UNet
from simple_diffusion.utils import save_images
from simple_diffusion.dataset import MultiFrameTIFDataset
from simple_diffusion.ema import EMA

warnings.filterwarnings("ignore", category=UserWarning,
                        message="Unable to find acceptable character detection dependency")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

n_timesteps = 1000
n_inference_timesteps = 250


def format_time(seconds):
    """将秒数格式化为 H小时 M分钟 S秒"""
    if seconds is None: return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def main(config):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(config.output_dir), exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)

    model = UNet(3, image_size=config.image_size, hidden_dims=[64, 128, 256, 512],
                 use_flash_attn=config.use_flash_attn).to(device)
    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps, beta_schedule="cosine")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.adam_weight_decay
    )

    global_step, checkpoint = 0, None
    if config.pretrained_model_path and os.path.exists(config.pretrained_model_path):
        print(f"从检查点恢复训练: {config.pretrained_model_path}")
        checkpoint = torch.load(config.pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state'])
        global_step = checkpoint.get('global_step', 0)
        print(f"将从 Global Step {global_step} 继续训练。")

    print(f"正在加载数据集于: {config.data_path}")
    train_dataset = MultiFrameTIFDataset(
        data_root=os.path.join(config.data_path, 'train'), frames_per_file=config.frames_per_file
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # --- 2. 使用 cycle() 让数据加载器可以无限循环 ---
    # 这样我们就不再受限于 epoch 的概念
    train_dataloader_cycle = cycle(train_dataloader)

    try:
        val_dataset = MultiFrameTIFDataset(
            data_root=os.path.join(config.data_path, 'val'), frames_per_file=config.frames_per_file
        )
        val_dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)
        fixed_val_batch = next(iter(val_dataloader))
    except (FileNotFoundError, StopIteration, ValueError):
        print("验证数据未找到或为空。将从训练集中取样用于可视化。")
        fixed_val_batch = next(iter(train_dataloader))
    fixed_val_low_res = fixed_val_batch["low_res"].to(device)

    # --- 3. 初始化组件时使用 max_train_steps ---
    ema = EMA(model, config.gamma, config.max_train_steps)
    if checkpoint and 'ema_model_state' in checkpoint:
        ema.ema_model.load_state_dict(checkpoint['ema_model_state'])

    lr_scheduler = get_scheduler(
        config.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps, num_training_steps=config.max_train_steps
    )
    for _ in range(global_step): lr_scheduler.step()

    summary(model, [(1, 3, config.image_size, config.image_size), (1,), (1, 3, 64, 64)], verbose=1)
    scaler = GradScaler(enabled=config.fp16_precision)

    # --- 4. 训练循环改为以步数为中心 ---
    print(f"\n--- 开始训练，总步数: {config.max_train_steps} ---")
    progress_bar = tqdm(initial=global_step, total=config.max_train_steps, desc="Steps")

    # 循环直到达到总步数
    while global_step < config.max_train_steps:
        batch = next(train_dataloader_cycle)  # 从无限循环的数据加载器中获取下一个批次

        high_res_images, low_res_images = batch["high_res"].to(device), batch["low_res"].to(device)
        batch_size = high_res_images.shape[0]
        noise = torch.randn(high_res_images.shape, device=device)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=device).long()
        noisy_images = noise_scheduler.add_noise(high_res_images, noise, timesteps)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=config.fp16_precision):
            noise_pred = model(noisy_images, timesteps, condition=low_res_images)["sample"]
            loss = F.l1_loss(noise_pred, noise)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        gamma = ema.update_gamma(global_step)
        ema.update_params(gamma)
        if config.use_clip_grad: clip_grad_norm_(model.parameters(), 1.0)
        lr_scheduler.step()

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        progress_bar.update(1)
        global_step += 1

        if global_step > 0 and global_step % config.save_model_steps == 0:
            print(f"\n--- Global Step {global_step}: 保存检查点并生成样本图像 ---")
            ema.ema_model.eval()
            with torch.no_grad():
                generator = torch.manual_seed(SEED)
                generated_images = noise_scheduler.generate(
                    ema.ema_model, num_inference_steps=n_inference_timesteps,
                    generator=generator, eta=1.0, batch_size=config.eval_batch_size,
                    condition=fixed_val_low_res
                )
                # epoch 参数不再有意义，可以传入 global_step
                save_images(generated_images, global_step, config)
                print(f"样本图像已保存到 '{config.samples_dir}' 目录中。")
                torch.save(
                    {'global_step': global_step, 'model_state': model.state_dict(),
                     'ema_model_state': ema.ema_model.state_dict(), 'optimizer_state': optimizer.state_dict()},
                    config.output_dir
                )
                print(f"模型检查点已保存到: {config.output_dir}")
            ema.ema_model.train()
            print("--- 已恢复训练模式 ---")

    progress_bar.close()

    # --- 5. 最终统计和预计时间功能 ---
    end_time = time.time()
    total_training_time = end_time - start_time
    avg_time_per_step = total_training_time / (global_step or 1)  # 避免除以零

    print("\n===================================================")
    print("               训练完成!                     ")
    print(f"总耗时: {format_time(total_training_time)}")
    print(f"平均每步耗时: {avg_time_per_step:.4f} 秒")
    print(f"最终模型保存在: {config.output_dir}")
    print("---------------------------------------------------")
    # 模拟一个新任务的预计时间
    example_steps = 100000
    estimated_time = avg_time_per_step * example_steps
    print(f"基于此速度，训练 {example_steps} 步预计需要: {format_time(estimated_time)}")
    print("===================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)
    main(config)