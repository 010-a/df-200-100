# train.py (最终完整版)

import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import yaml
from types import SimpleNamespace
import warnings

# --- 导入所有必要的库 ---
from diffusers.optimization import get_scheduler
from torchinfo import summary
from simple_diffusion.scheduler import DDIMScheduler
from simple_diffusion.model import UNet
from simple_diffusion.utils import save_images
from simple_diffusion.dataset import MultiFrameTIFDataset
from simple_diffusion.ema import EMA

# --- 抑制底层库的警告 (可选) ---
warnings.filterwarnings("ignore", category=UserWarning,
                        message="Unable to find acceptable character detection dependency")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

n_timesteps = 1000
n_inference_timesteps = 250


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(config.output_dir), exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)

    # --- 1. 初始化模型和优化器 ---
    model = UNet(3, image_size=config.image_size, hidden_dims=[64, 128, 256, 512],
                 use_flash_attn=config.use_flash_attn)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
    )

    start_epoch = 0
    global_step = 0

    # --- 2. 检查并从断点恢复 ---
    if config.pretrained_model_path:
        print(f"从检查点恢复训练: {config.pretrained_model_path}")
        checkpoint = torch.load(config.pretrained_model_path, map_location=device)

        model.load_state_dict(checkpoint['model_state'])

        # 只有在检查点中存在时才加载优化器状态，以保持向后兼容性
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)

        print(f"将从 Epoch {start_epoch} 和 Global Step {global_step} 继续训练。")

    # --- 3. 数据加载 ---
    print(f"正在加载数据集于: {config.data_path}")
    train_dataset = MultiFrameTIFDataset(
        data_root=os.path.join(config.data_path, 'train'),
        frames_per_file=config.frames_per_file
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    try:
        val_dataset = MultiFrameTIFDataset(
            data_root=os.path.join(config.data_path, 'val'),
            frames_per_file=config.frames_per_file
        )
        val_dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)
        fixed_val_batch = next(iter(val_dataloader))
    except (FileNotFoundError, StopIteration):
        print("验证数据未找到或为空。将从训练集中取样用于可视化。")
        fixed_val_batch = next(iter(train_dataloader))

    fixed_val_low_res = fixed_val_batch["low_res"].to(device)

    # --- 4. 初始化 EMA 和学习率调度器 ---
    steps_per_epoch = len(train_dataloader)
    total_num_steps = (steps_per_epoch * config.num_epochs) // config.gradient_accumulation_steps

    ema = EMA(model, config.gamma, total_num_steps)
    if config.pretrained_model_path and 'ema_model_state' in checkpoint:
        ema.ema_model.load_state_dict(checkpoint['ema_model_state'])

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=total_num_steps,
    )

    # 将调度器快进到正确的步数以恢复状态
    for _ in range(global_step):
        lr_scheduler.step()

    summary(model, [(1, 3, config.image_size, config.image_size), (1,), (1, 3, 64, 64)], verbose=1)
    scaler = GradScaler(enabled=config.fp16_precision)

    # --- 5. 训练循环 ---
    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}", initial=0, leave=True)

        for step, batch in enumerate(train_dataloader):
            high_res_images = batch["high_res"].to(device)
            low_res_images = batch["low_res"].to(device)

            batch_size = high_res_images.shape[0]
            noise = torch.randn(high_res_images.shape, device=device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=device).long()
            noisy_images = noise_scheduler.add_noise(high_res_images, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=config.fp16_precision):
                noise_pred = model(noisy_images, timesteps, condition=low_res_images)["sample"]
                loss = F.l1_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            gamma = ema.update_gamma(global_step)
            ema.update_params(gamma)

            if config.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)

            lr_scheduler.step()
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

            if global_step > 0 and global_step % config.save_model_steps == 0:
                ema.ema_model.eval()
                with torch.no_grad():
                    generator = torch.manual_seed(SEED)
                    generated_images = noise_scheduler.generate(
                        ema.ema_model, num_inference_steps=n_inference_timesteps,
                        generator=generator, eta=1.0, batch_size=config.eval_batch_size,
                        condition=fixed_val_low_res, show_progress=True
                    )
                    save_images(generated_images, epoch, config)

                    # 保存完整的检查点
                    torch.save(
                        {'epoch': epoch, 'global_step': global_step, 'model_state': model.state_dict(),
                         'ema_model_state': ema.ema_model.state_dict(), 'optimizer_state': optimizer.state_dict()},
                        config.output_dir
                    )
                ema.ema_model.train()

        progress_bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    config = SimpleNamespace(**config_dict)

    main(config)