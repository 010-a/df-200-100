# train.py (最终完整修复版 v3 - 修复 GradScaler TypeError)

import argparse, numpy as np, random, torch, torch.nn.functional as F, os, yaml, warnings, time
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast  # <-- 直接从这里导入
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from types import SimpleNamespace
from itertools import cycle
from pathlib import Path
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from diffusers.optimization import get_scheduler
from torchinfo import summary
from simple_diffusion.scheduler import DDIMScheduler
from simple_diffusion.model import UNet
from simple_diffusion.utils import save_images
from simple_diffusion.dataset import MultiFrameTIFDataset
from simple_diffusion.ema import EMA

warnings.filterwarnings("ignore", category=UserWarning,
                        message="Unable to find acceptable character detection dependency")
SEED = 42;
random.seed(SEED);
np.random.seed(SEED);
torch.manual_seed(SEED);
torch.cuda.manual_seed(SEED)
n_timesteps = 1000;
n_inference_timesteps = 100


def format_time(seconds):
    if seconds is None: return "N/A"; h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(
        seconds % 60); return f"{h}h {m}m {s}s"


def validate_and_save(model, noise_scheduler, val_dataloader, device, config, global_step, best_metrics,
                      optimizer_state):
    model.eval()
    psnr_metric, ssim_metric = PeakSignalNoiseRatio(data_range=1.0).to(device), StructuralSimilarityIndexMeasure(
        data_range=1.0).to(device)
    print(f"\n--- Global Step {global_step}: 开始验证... ---")
    num_val_batches, last_generated_output = getattr(config, 'num_validation_batches', 16), None
    with torch.no_grad():
        for i, batch in enumerate(
                tqdm(val_dataloader, desc="Validation", leave=False, total=min(len(val_dataloader), num_val_batches))):
            if i >= num_val_batches: break
            low_res_batch, high_res_batch = batch["low_res"].to(device), batch["high_res"].to(device)
            generated_output = noise_scheduler.generate(model, num_inference_steps=n_inference_timesteps,
                                                        generator=torch.manual_seed(SEED),
                                                        batch_size=low_res_batch.shape[0], condition=low_res_batch)
            pred_images, target_images = generated_output["sample_pt"].to(device), (high_res_batch + 1.0) / 2.0
            if pred_images.shape != target_images.shape: print(
                f"\n[警告] 验证步骤中检测到维度不匹配，已跳过此批次。 Pred: {pred_images.shape}, Target: {target_images.shape}"); continue
            psnr_metric.update(pred_images, target_images);
            ssim_metric.update(pred_images, target_images);
            last_generated_output = generated_output
    if psnr_metric.total == 0: print("\n[错误] 未能计算任何验证指标。"); model.train(); return best_metrics
    avg_psnr, avg_ssim = psnr_metric.compute().item(), ssim_metric.compute().item()
    print(f"--- 验证完成 --- PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} ---")

    output_dir = Path(config.output_dir)
    latest_checkpoint_path = output_dir / config.output_filename
    best_model_path = output_dir / "best_model.pth"

    checkpoint_payload = {'global_step': global_step, 'model_state': model.state_dict(),
                          'optimizer_state': optimizer_state}
    torch.save(checkpoint_payload, latest_checkpoint_path);
    print(f"最新检查点已保存到: {latest_checkpoint_path}")
    if avg_psnr > best_metrics["psnr"]:
        best_metrics.update({"psnr": avg_psnr, "ssim": avg_ssim, "step": global_step})
        best_payload = {**checkpoint_payload, 'psnr': avg_psnr, 'ssim': avg_ssim};
        torch.save(best_payload, best_model_path)
        print(f"🎉 新的最佳模型! PSNR: {avg_psnr:.4f}. 已保存到: {best_model_path}")
    if last_generated_output: save_images(last_generated_output, global_step, config); print(
        f"样本图像已保存到 '{config.samples_dir}' 目录中。")
    model.train();
    print("--- 已恢复训练模式 ---");
    return best_metrics


def main(config):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)

    model = UNet(3, image_size=config.image_size, hidden_dims=[64, 128, 256, 512],
                 use_flash_attn=config.use_flash_attn).to(device)
    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps, beta_schedule="cosine")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.adam_weight_decay)
    global_step, checkpoint = 0, None
    if config.pretrained_model_path and os.path.exists(config.pretrained_model_path):
        print(f"从检查点恢复训练: {config.pretrained_model_path}");
        checkpoint = torch.load(config.pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state']);
        if 'optimizer_state' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state'])
        global_step = checkpoint.get('global_step', 0);
        print(f"将从 Global Step {global_step} 继续训练。")
    print(f"正在加载数据集于: {config.data_path}")
    train_dataset = MultiFrameTIFDataset(data_root=os.path.join(config.data_path, 'train'),
                                         frames_per_file=config.frames_per_file)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
    train_dataloader_cycle = cycle(train_dataloader)
    val_dataset = MultiFrameTIFDataset(data_root=os.path.join(config.data_path, 'val'),
                                       frames_per_file=config.frames_per_file)
    val_dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)
    ema = EMA(model, config.gamma, config.max_train_steps)
    if checkpoint and 'ema_model_state' in checkpoint: ema.ema_model.load_state_dict(checkpoint['ema_model_state'])
    lr_scheduler = get_scheduler(config.lr_scheduler, optimizer=optimizer, num_warmup_steps=config.lr_warmup_steps,
                                 num_training_steps=config.max_train_steps)
    for _ in range(global_step): lr_scheduler.step()
    summary(model, [(1, 3, config.image_size, config.image_size), (1,), (1, 3, 64, 64)], verbose=1)

    # --- [核心修复] ---
    # 恢复到您之前能正常运行的旧版 API 写法
    scaler = GradScaler(enabled=config.fp16_precision)
    # --- 修复结束 ---

    print(f"\n--- 开始训练，总步数: {config.max_train_steps} ---")
    progress_bar = tqdm(initial=global_step, total=config.max_train_steps, desc="Steps")
    best_metrics = {"psnr": -1.0, "ssim": -1.0, "step": -1}
    while global_step < config.max_train_steps:
        batch = next(train_dataloader_cycle)
        high_res_images, low_res_images = batch["high_res"].to(device), batch["low_res"].to(device)
        optimizer.zero_grad(set_to_none=True)

        # --- [核心修复] ---
        # 恢复到您之前能正常运行的旧版 API 写法
        with autocast(enabled=config.fp16_precision):
            # --- 修复结束 ---
            noise = torch.randn(high_res_images.shape, device=device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (high_res_images.shape[0],),
                                      device=device).long()
            noisy_images = noise_scheduler.add_noise(high_res_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps, condition=low_res_images)["sample"];
            loss = F.l1_loss(noise_pred, noise)

        scaler.scale(loss).backward();
        scaler.step(optimizer);
        scaler.update()
        gamma = ema.update_gamma(global_step);
        ema.update_params(gamma)
        if config.use_clip_grad: clip_grad_norm_(model.parameters(), 1.0)
        lr_scheduler.step();
        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]};
        progress_bar.set_postfix(**logs)
        global_step += 1

        if global_step > 0 and global_step % config.save_model_steps == 0:
            best_metrics = validate_and_save(ema.ema_model, noise_scheduler, val_dataloader, device, config,
                                             global_step, best_metrics, optimizer.state_dict())

    progress_bar.close()
    print("\n===================================================")
    print("               训练完成!                     ")
    print(f"总耗时: {format_time(time.time() - start_time)}");
    print("--- 历史最佳模型 ---")
    print(f"Step: {best_metrics['step']}");
    print(f"PSNR: {best_metrics['psnr']:.4f}");
    print(f"SSIM: {best_metrics['ssim']:.4f}")
    best_model_path = Path(config.output_dir) / "best_model.pth"
    print(f"最佳模型保存在: {best_model_path}")
    print("===================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f: config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)
    main(config)