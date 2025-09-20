# train.py (最终版本)

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
import yaml  # <-- 新增：用于读取YAML文件
from types import SimpleNamespace  # <-- 新增：用于将字典转换为对象

# --- [修复] 补全缺失的import ---
from diffusers.optimization import get_scheduler
from torchinfo import summary
# -----------------------------

from simple_diffusion.scheduler import DDIMScheduler
from simple_diffusion.model import UNet
from simple_diffusion.utils import save_images
from simple_diffusion.dataset import MultiFrameTIFDataset
from simple_diffusion.ema import EMA

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

n_timesteps = 1000
n_inference_timesteps = 250


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保输出和样本目录存在
    os.makedirs(os.path.dirname(config.output_dir), exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)

    # --- 1. 模型初始化 ---
    model = UNet(3, image_size=config.image_size, hidden_dims=[64, 128, 256, 512],
                 use_flash_attn=config.use_flash_attn)
    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")
    model = model.to(device)

    if config.pretrained_model_path:
        print(f"Loading pretrained model from: {config.pretrained_model_path}")
        pretrained = torch.load(config.pretrained_model_path)["model_state"]
        model.load_state_dict(pretrained)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
    )

    # --- 2. 数据加载 ---
    print(f"Loading dataset from: {config.data_path}")
    train_dataset = MultiFrameTIFDataset(
        data_root=os.path.join(config.data_path, 'train'),
        frames_per_file=config.frames_per_file
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4
    )

    try:
        val_dataset = MultiFrameTIFDataset(
            data_root=os.path.join(config.data_path, 'val'),
            frames_per_file=config.frames_per_file
        )
        val_dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)
        fixed_val_batch = next(iter(val_dataloader))
    except (FileNotFoundError, StopIteration):
        print("Validation data not found or empty. Using a batch from training set for sampling.")
        fixed_val_batch = next(iter(train_dataloader))

    fixed_val_low_res = fixed_val_batch["low_res"].to(device)

    # --- 3. 训练准备 ---
    steps_per_epoch = len(train_dataloader)
    total_num_steps = (steps_per_epoch * config.num_epochs) // config.gradient_accumulation_steps
    total_num_steps += int(total_num_steps * 10 / 100)

    gamma = config.gamma
    ema = EMA(model, gamma, total_num_steps)

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=total_num_steps,
    )

    summary(model, [(1, 3, config.image_size, config.image_size), (1,), (1, 3, 64, 64)], verbose=1)

    scaler = GradScaler(enabled=config.fp16_precision)
    global_step = 0

    # --- 4. 训练循环 ---
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            high_res_images = batch["high_res"].to(device)
            low_res_images = batch["low_res"].to(device)

            batch_size = high_res_images.shape[0]
            noise = torch.randn(high_res_images.shape, device=device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=device).long()
            noisy_images = noise_scheduler.add_noise(high_res_images, noise, timesteps)

            optimizer.zero_grad()
            with autocast(enabled=config.fp16_precision):
                noise_pred = model(noisy_images, timesteps, condition=low_res_images)["sample"]
                loss = F.l1_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema.update_params(gamma)
            gamma = ema.update_gamma(global_step)

            if config.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)

            lr_scheduler.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            global_step += 1

            if global_step > 0 and global_step % config.save_model_steps == 0:
                ema.ema_model.eval()
                with torch.no_grad():
                    generator = torch.manual_seed(SEED)
                    generated_images = noise_scheduler.generate(
                        ema.ema_model,
                        num_inference_steps=n_inference_timesteps,
                        generator=generator, eta=1.0,
                        batch_size=config.eval_batch_size,
                        condition=fixed_val_low_res
                    )

                    save_images(generated_images, epoch, config)

                    torch.save(
                        {'model_state': model.state_dict(), 'ema_model_state': ema.ema_model.state_dict(),
                         'optimizer_state': optimizer.state_dict()},
                        config.output_dir
                    )
                ema.ema_model.train()

        progress_bar.close()


if __name__ == "__main__":
    # --- [修改] 使用YAML文件加载配置 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    with open(args.config, 'r',encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 将字典转换为对象，方便使用 config.parameter 的形式访问
    config = SimpleNamespace(**config_dict)

    main(config)