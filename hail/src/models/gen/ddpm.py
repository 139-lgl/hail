"""
DDPM (Denoising Diffusion Probabilistic Models) 扩散模型实现

这个模块实现了经典的DDPM扩散模型，具有以下特点：
1. 基于马尔可夫链的前向扩散过程
2. 反向去噪过程用于生成
3. 支持DDPM和DDIM两种采样方法
4. 支持条件生成（基于历史帧和动作）

DDPM是扩散模型的经典实现，为后续的改进版本（如EDM）奠定了基础。

参考论文：Denoising Diffusion Probabilistic Models (Ho et al., 2020)
参考论文：Denoising Diffusion Implicit Models (Song et al., 2020)

作者：Snake Diffusion项目组
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import importlib.util

from .blocks import BaseDiffusionModel, UNetConfig
from .edm import VAEAdapter, ActionChange
from .edm import VAEAdapter

@dataclass
class DDPMConfig:
    """
    DDPM模型配置类
    
    包含DDPM模型的基本配置参数
    
    Attributes:
        T: 扩散步数（时间步总数）
        unet: UNet网络配置
    """
    T: int                    # 扩散时间步数
    unet: UNetConfig         # UNet网络配置

class DDPM(BaseDiffusionModel):
    """
    DDPM (Denoising Diffusion Probabilistic Models) 扩散模型
    
    这是经典的扩散模型实现，包含以下核心组件：
    1. 前向扩散过程：逐步向图像添加高斯噪声
    2. 反向去噪过程：通过神经网络逐步去除噪声
    3. 支持条件生成：基于历史帧和动作序列
    4. 两种采样方法：DDPM（慢但高质量）和DDIM（快速采样）
    """
    
    @classmethod
    def from_config(
        cls,
        config: DDPMConfig,
        context_length: int,
        device: str,
        model: nn.Module,
        use_latent_encoder: bool = False
    ):
        """
        从配置创建DDPM实例
        
        Args:
            config: DDPM配置
            context_length: 上下文长度（历史帧数量）
            device: 计算设备
            model: 底层UNet模型
            
        Returns:
            DDPM实例
        """
        return cls(
            T=config.T,
            context_length=context_length,
            device=device,
            model=model,
            use_latent_encoder=use_latent_encoder
        )

    def __init__(
        self,
        T: int,
        context_length: int,
        device: str,
        model: nn.Module,
        use_latent_encoder: bool = False
    ):
        """
        初始化DDPM模型
        
        Args:
            T: 扩散时间步数
            context_length: 历史帧数量
            device: 计算设备
            model: 去噪网络（UNet）
        """
        super().__init__()
        self.T = T                                    # 扩散时间步数
        self.eps_model = model.to(device)            # 噪声预测网络
        self.device = device
        self.context_length = context_length         # 历史帧数量
        self.vae = VAEAdapter(device=device) if use_latent_encoder else None
        if self.vae is not None:
            self.vae._try_init()
        self.use_latent = bool(use_latent_encoder and self.vae is not None and self.vae.available)
        self.action_change = ActionChange(device=device)
        
        # 定义噪声调度（beta schedule）
        # 从小到大的线性调度，控制每个时间步添加的噪声量
        beta_schedule = torch.linspace(1e-4, 0.02, T + 1, device=self.device)
        
        # 计算alpha参数：alpha_t = 1 - beta_t
        alpha_t_schedule = 1 - beta_schedule
        
        # 计算累积alpha：bar_alpha_t = ∏(alpha_s) for s=1 to t
        # 这个参数决定了从原始图像到时间步t的总噪声量
        bar_alpha_t_schedule = torch.cumprod(alpha_t_schedule.detach().cpu(), 0).to(self.device)
        
        # 预计算常用的平方根值，用于加速前向和反向过程
        sqrt_bar_alpha_t_schedule = torch.sqrt(bar_alpha_t_schedule)
        sqrt_minus_bar_alpha_t_schedule = torch.sqrt(1 - bar_alpha_t_schedule)
        
        # 将这些参数注册为缓冲区，确保它们随模型一起保存和加载
        self.register_buffer("beta_schedule", beta_schedule)
        self.register_buffer("alpha_t_schedule", alpha_t_schedule)
        self.register_buffer("bar_alpha_t_schedule", bar_alpha_t_schedule)
        self.register_buffer("sqrt_bar_alpha_t_schedule", sqrt_bar_alpha_t_schedule)
        self.register_buffer("sqrt_minus_bar_alpha_t_schedule", sqrt_minus_bar_alpha_t_schedule)
        
        # 损失函数：均方误差，用于比较预测噪声和真实噪声
        self.criterion = nn.MSELoss()        

    def forward(self, imgs: torch.Tensor, prev_frames: torch.Tensor, prev_actions: torch.Tensor):
        """
        训练时的前向传播
        
        实现DDPM的训练过程：
        1. 随机选择时间步t
        2. 向图像添加噪声（前向扩散）
        3. 预测添加的噪声
        4. 计算预测噪声与真实噪声的MSE损失
        
        Args:
            imgs: 目标图像 [B, C, H, W]
            prev_frames: 历史帧序列 [B, context_length, C, H, W]
            prev_actions: 历史动作序列 [B, context_length, ...]
            
        Returns:
            训练损失（标量）
        """
        assert prev_frames.shape[1] == prev_actions.shape[1] == self.context_length
        
        # 随机采样时间步t（从1到T）
        t = torch.randint(low=1, high=self.T+1, size=(imgs.shape[0],), device=self.device)
        
        # 生成随机噪声
        if self.use_latent and self.vae and self.vae.available:
            y = self.vae.encode_image(imgs)
            pf = self.vae.encode_frames(prev_frames)
            noise = torch.randn_like(y, device=self.device)
            batch_size = y.shape[0]
            noise_imgs = self.sqrt_bar_alpha_t_schedule[t].view((batch_size, 1, 1 ,1)) * y \
                + self.sqrt_minus_bar_alpha_t_schedule[t].view((batch_size, 1, 1, 1)) * noise
            if pf.dim() == 5:
                big = torch.concat([noise_imgs[:, None, :, :, :], pf], dim=1)
            else:
                b, n, k, c, h, w = pf.shape
                pf = pf.view(b, n * k, c, h, w)
                big = torch.concat([noise_imgs[:, None, :, :, :], pf], dim=1)
            noise_imgs = big.flatten(1,2)
            cond_actions = self.action_change(prev_actions)
            pred_noise = self.eps_model(noise_imgs, t.unsqueeze(1), cond_actions)
            return self.criterion(pred_noise, noise)
        noise = torch.randn_like(imgs, device=self.device)
        batch_size, channels, width, height = imgs.shape
        
        # 前向扩散过程：根据时间步t添加噪声
        # 使用重参数化技巧：x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1-bar_alpha_t) * epsilon
        noise_imgs = self.sqrt_bar_alpha_t_schedule[t].view((batch_size, 1, 1 ,1)) * imgs \
            + self.sqrt_minus_bar_alpha_t_schedule[t].view((batch_size, 1, 1, 1)) * noise
        if prev_frames.dim() == 5:
            big = torch.concat([noise_imgs[:, None, :, :, :], prev_frames], dim=1)
        else:
            b, n, k, c, h, w = prev_frames.shape
            pf = prev_frames.view(b, n * k, c, h, w)
            big = torch.concat([noise_imgs[:, None, :, :, :], pf], dim=1)
        noise_imgs = big.flatten(1,2)

        # 通过网络预测噪声
        cond_actions = self.action_change(prev_actions)
        pred_noise = self.eps_model(noise_imgs, t.unsqueeze(1), cond_actions)

        # 计算预测噪声与真实噪声的MSE损失
        return self.criterion(pred_noise, noise)
    
    @torch.no_grad()
    def sample_ddpm(
        self,
        size: Tuple[int],
        prev_frames: torch.Tensor,
        prev_actions: torch.Tensor
    ):
        """
        DDPM采样方法（原始方法）
        
        使用完整的反向马尔可夫链进行采样，质量高但速度慢。
        从纯噪声开始，逐步去噪直到生成清晰图像。
        
        Args:
            size: 生成图像的尺寸 (C, H, W)
            prev_frames: 历史帧序列
            prev_actions: 历史动作序列
            
        Returns:
            生成的图像
        """
        # 从纯噪声开始
        if self.use_latent and self.vae and self.vae.available:
            c, h, w = size
            h = h // 8
            w = w // 8
            x_t = torch.randn(1, 4, h, w, device=self.device)
        else:
            x_t = torch.randn(1, *size, device=self.device)
        
        # 反向扩散过程：从t=T到t=1
        for t in range(self.T, 0, -1):
            # 除了最后一步，都需要添加随机噪声
            z = torch.randn_like(x_t, device=self.device) if t > 1 else 0
            
            # 准备时间步张量
            t_tensor = torch.tensor([t], device=self.device).repeat(x_t.shape[0], 1)
            
            # 将当前图像与历史帧拼接
            if self.use_latent and self.vae and self.vae.available:
                pf = self.vae.encode_frames(prev_frames)
                if pf.dim() == 5:
                    big_x_t = torch.concat([x_t[:, None, :, :, :], pf], dim=1)
                else:
                    b, n, k, c, h, w = pf.shape
                    pf = pf.view(b, n * k, c, h, w)
                    big_x_t = torch.concat([x_t[:, None, :, :, :], pf], dim=1)
                big_x_t = big_x_t.flatten(1,2)
            else:
                if prev_frames.dim() == 5:
                    big_x_t = torch.concat([x_t[:, None, :, :, :], prev_frames], dim=1)
                else:
                    b, n, k, c, h, w = prev_frames.shape
                    pf = prev_frames.view(b, n * k, c, h, w)
                    big_x_t = torch.concat([x_t[:, None, :, :, :], pf], dim=1)
                big_x_t = big_x_t.flatten(1,2)
            
            # 预测噪声
            cond_actions = self.action_change(prev_actions)
            pred_noise = self.eps_model(big_x_t, t_tensor, cond_actions)
            
            # DDPM反向更新公式
            x_t = 1 / torch.sqrt(self.alpha_t_schedule[t]) * \
                (x_t - pred_noise * (1 - self.alpha_t_schedule[t]) / self.sqrt_minus_bar_alpha_t_schedule[t]) + \
                torch.sqrt(self.beta_schedule[t]) * z
        if self.use_latent and self.vae and self.vae.available:
            img = self.vae.decode_latents(x_t)
            return img
        return x_t
        
    @torch.no_grad()
    def sample(
        self, steps: int, size: Tuple[int],
        prev_frames: torch.Tensor,
        prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        DDIM采样方法（快速采样）
        
        使用确定性的DDIM采样算法，可以用更少的步数生成高质量图像。
        通过跳过某些时间步来加速采样过程。
        
        Args:
            steps: 采样步数（可以远小于训练时的T）
            size: 生成图像的尺寸 (C, H, W)
            prev_frames: 历史帧序列
            prev_actions: 历史动作序列
            
        Returns:
            生成的图像
        """
        # 从纯噪声开始
        if self.use_latent and self.vae and self.vae.available:
            c, h, w = size
            h = h // 8
            w = w // 8
            x_t = torch.randn(1, 4, h, w, device=self.device)
        else:
            x_t = torch.randn(1, *size, device=self.device)
        
        # 计算时间步跳跃间隔
        step_size = self.T // steps
        
        # 创建时间步序列（从T到0，步长为step_size）
        range_t = range(self.T, -1, -step_size)
        next_range_t = range_t[1:]
        range_t = range_t[:-1]
        
        # DDIM采样循环
        for i, j in zip(range_t, next_range_t):
            # 准备当前时间步
            t_tensor = torch.tensor([i], device=self.device).repeat(x_t.shape[0], 1)

            # 将当前图像与历史帧拼接
            if self.use_latent and self.vae and self.vae.available:
                pf = self.vae.encode_frames(prev_frames)
                if pf.dim() == 5:
                    big_x_t = torch.concat([x_t[:, None, :, :, :], pf], dim=1)
                else:
                    b, n, k, c, h, w = pf.shape
                    pf = pf.view(b, n * k, c, h, w)
                    big_x_t = torch.concat([x_t[:, None, :, :, :], pf], dim=1)
                big_x_t = big_x_t.flatten(1,2)
            else:
                if prev_frames.dim() == 5:
                    big_x_t = torch.concat([x_t[:, None, :, :, :], prev_frames], dim=1)
                else:
                    b, n, k, c, h, w = prev_frames.shape
                    pf = prev_frames.view(b, n * k, c, h, w)
                    big_x_t = torch.concat([x_t[:, None, :, :, :], pf], dim=1)
                big_x_t = big_x_t.flatten(1,2)
            
            # 预测噪声
            cond_actions = self.action_change(prev_actions)
            pred_noise = self.eps_model(big_x_t, t_tensor, cond_actions)
            
            # DDIM更新公式
            alpha = self.bar_alpha_t_schedule[i]        # 当前时间步的alpha
            next_alpha = self.bar_alpha_t_schedule[j]   # 下一时间步的alpha
            
            # 预测原始图像x_0
            x0 = (x_t - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            
            # 计算下一时间步的噪声部分
            new_xt = torch.sqrt(1 - next_alpha) * pred_noise

            # 更新到下一时间步
            x_t = torch.sqrt(next_alpha) * x0 + new_xt
        if self.use_latent and self.vae and self.vae.available:
            img = self.vae.decode_latents(x_t)
            return img
        return x_t
        
if __name__ == "__main__":
    """
    测试代码
    
    验证DDPM模型的基本功能
    """
    # 测试参数
    size = (64, 64)          # 图像尺寸
    input_channels = 3       # RGB通道
    context_length = 4       # 历史帧数量
    actions_count = 5        # 动作类别数
    T = 1000                # 扩散时间步数
    batch_size = 3          # 批次大小

    from blocks import UNet

    # 创建UNet网络
    unet = UNet((input_channels) * (context_length + 1), 3, T, actions_count, context_length)
    
    # 创建DDPM模型
    ddpm = DDPM(
        T=T,
        model=unet,
        context_length=context_length,
        device="cpu"
    )

    # 准备测试数据
    img = torch.randn((batch_size, input_channels, *size))  # 目标图像
    prev_frames = torch.randn((batch_size, context_length, input_channels, *size))  # 历史帧
    prev_actions = torch.randint(low=0, high=actions_count, size=(batch_size, context_length))  # 历史动作
    
    # 测试前向传播（训练损失计算）
    loss = ddpm.forward(img, prev_frames, prev_actions)
    print(f"Training loss: {loss.item()}")
