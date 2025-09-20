# simple_diffusion/scheduler/ddim.py (最终完整版)

# Disclaimer: This code was influenced by
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py

import math
import numpy as np
import torch
from tqdm import tqdm

from simple_diffusion.utils import unnormalize_to_zero_to_one


def cosine_beta_schedule(timesteps, beta_start=0.0, beta_end=0.999, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end)


class DDIMScheduler:
    def __init__(
            self,
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="cosine",
            clip_sample=True,
            set_alpha_to_one=True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.betas = (
            cosine_beta_schedule(num_train_timesteps, beta_start, beta_end)
            if beta_schedule == "cosine"
            else torch.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=torch.float32
            )
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )
        self.timesteps = np.arange(num_train_timesteps)[::-1]

    def _set_timesteps(self, num_inference_steps, offset=0):
        self.timesteps = (
                np.arange(
                    0,
                    self.num_train_timesteps,
                    self.num_train_timesteps // num_inference_steps,
                )[::-1] + offset
        )

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        return (
                (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        )

    def _step(self, model_output, timestep, sample, eta=1.0, generator=None):
        prev_timestep = timestep - self.num_train_timesteps // len(self.timesteps)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** 0.5
        model_output = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        if eta > 0:
            noise = torch.randn(model_output.shape, generator=generator).to(sample.device)
            prev_sample += std_dev_t * noise

        return prev_sample

    def add_noise(self, original_samples, noise, timesteps):
        self.alphas_cumprod = self.alphas_cumprod.to(timesteps.device)
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        return (
                sqrt_alpha_prod[:, None, None, None] * original_samples
                + sqrt_one_minus_alpha_prod[:, None, None, None] * noise
        )

    @torch.no_grad()
    def generate(
            self,
            model,
            batch_size=1,
            generator=None,
            eta=1.0,
            num_inference_steps=50,
            device=None,
            condition=None,
            show_progress=True,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        image = torch.randn(
            (batch_size, model.in_channels, model.sample_size, model.sample_size),
            generator=generator,
        ).to(device)
        self._set_timesteps(num_inference_steps)

        # 使用 leave=False 可以在进度条完成后将其从终端清除
        progress_bar = tqdm(self.timesteps, disable=not show_progress, desc="Generating sample", leave=False)
        for t in progress_bar:
            model_output = model(image, t, condition=condition)["sample"]
            image = self._step(model_output, t, image, eta, generator=generator)

        image = unnormalize_to_zero_to_one(image)
        return {"sample": image.cpu().permute(0, 2, 3, 1).numpy(), "sample_pt": image}

    def __len__(self):
        return self.num_train_timesteps