"""
EDM (Elucidating the Design Space of Diffusion-Based Generative Models) 扩散模型实现

这个模块实现了EDM扩散模型，这是一种先进的扩散模型架构，具有以下特点：
1. 改进的噪声调度策略
2. 优化的损失函数设计
3. 更好的采样算法
4. 支持条件生成（基于历史帧和动作）

EDM相比传统DDPM有更好的生成质量和训练稳定性。

参考论文：Elucidating the Design Space of Diffusion-Based Generative Models
官方实现：https://github.com/NVlabs/edm

作者：Snake Diffusion项目组
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import importlib.util
import types
import os

from .blocks import BaseDiffusionModel, UNetConfig
from typing import Optional

@dataclass
class EDMConfig:
    """
    EDM模型配置类
    
    包含EDM模型的所有超参数配置
    
    Attributes:
        unet: UNet网络配置
        p_mean: 噪声分布的均值参数
        p_std: 噪声分布的标准差参数
        sigma_data: 数据的标准差，用于归一化
        sigma_min: 最小噪声水平
        sigma_max: 最大噪声水平
        rho: 噪声调度的形状参数
    """
    unet: UNetConfig
    p_mean: float
    p_std: float
    sigma_data: float
    sigma_min: float = 0.002
    sigma_max: float = 80
    rho: float = 7

# 部分代码参考了EDM论文的官方实现
# https://github.com/NVlabs/edm

class ActionChange(nn.Module):
    """
    动作变化处理模块
    
    将输入的动作序列转换为适合模型使用的特征表示。
    采用分块处理的方式，将输入分成8x8的小块，每个块通过MLP处理成一个标量值。
    """
    
    def __init__(self, device: str):
        """
        初始化动作变化模块
        
        Args:
            device: 计算设备（cuda/cpu）
        """
        super().__init__()
        
        # 多层感知机：将每个8x8块的特征映射为单个数值
        # 输入维度：128 (2通道 × 8×8像素)
        # 输出维度：1 (每个块一个标量值)
        # 动态MLP：根据通道数构建输入维度 (C * 8 * 8)
        # 由于动作通道可扩展（风向、风速、海拔），在首次前向时初始化
        self.mlp: Optional[nn.Sequential] = None
        self._in_dim: Optional[int] = None
        self.patch_size: int = 8
        self.device = device

    def forward(self, x: torch.Tensor):
        """
        前向传播
        
        将输入张量分成8x8的块，每个块通过MLP处理成一个标量值
        
        Args:
            x: 输入张量 [B, N, C, H, W]
            
        Returns:
            处理后的特征向量 [B, N×64]
        """
        if x.dtype != torch.float32:
            x = x.float()
        B = x.shape[0]
        if x.dim() == 5:
            N = x.shape[1]
            x = x.unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)
            x = x.permute(0, 1, 3, 4, 2, 5, 6).contiguous()
            C = x.size(4)
            x = x.view(B, N, 64, C, self.patch_size, self.patch_size)
            x = x.view(B, N, 64, -1)
            in_dim = C * self.patch_size * self.patch_size
            if self.mlp is None:
                self._in_dim = in_dim
                self.mlp = nn.Sequential(
                    nn.Linear(in_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                ).to(self.device)
            x = self.mlp(x)
            x = x.squeeze(-1)
            x = x.view(x.size(0), -1)
            return x
        elif x.dim() == 6:
            N = x.shape[1]
            K = x.shape[2]
            x = x.unfold(4, self.patch_size, self.patch_size).unfold(5, self.patch_size, self.patch_size)
            x = x.permute(0, 1, 2, 4, 5, 3, 6, 7).contiguous()
            C = x.size(5)
            x = x.view(B, N, K, 64, C, self.patch_size, self.patch_size)
            x = x.view(B, N, K, 64, -1)
            in_dim = C * self.patch_size * self.patch_size + 2
            if self.mlp is None:
                self._in_dim = in_dim
                self.mlp = nn.Sequential(
                    nn.Linear(in_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                ).to(self.device)
            pos = torch.tensor([
                [0.0, 0.0],
                [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                [0.0, -1.0], [0.0, 1.0],
                [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]
            ], dtype=x.dtype, device=x.device)
            if K != 9:
                pos = pos[:K]
            pos = pos.view(1, 1, K, 1, 2).expand(B, N, K, 64, 2)
            x = torch.cat([x, pos], dim=-1)
            x = self.mlp(x)
            x = x.squeeze(-1)
            x = x.view(x.size(0), -1)
            return x
        else:
            raise ValueError("Unsupported actions tensor shape")

class CNEncoderAdapter(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.available = False
        self.encoder = None
        self._init_attempted = False
        self._preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _try_init(self):
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            spec = importlib.util.spec_from_file_location("cn_encoder_mod", "/mnt/sdb/lgl/EscherNet2/4DoF/CN_encoder.py")
            if spec is None or spec.loader is None:
                return
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            CN_encoder_cls = getattr(mod, "CN_encoder", None)
            if CN_encoder_cls is None:
                return
            model_root = "/mnt/sdb/lgl/EscherNet2/model/convnextv2-tiny-22k-224"
            self.encoder = CN_encoder_cls.from_pretrained(model_root)
            self.encoder.eval()
            self.encoder.to(self.device)
            self.available = True
        except Exception:
            self.available = False

    def forward(self, frames: torch.Tensor) -> Optional[torch.Tensor]:
        self._try_init()
        if not self.available or self.encoder is None:
            return None
        if frames.dim() == 5:
            b, n, c, h, w = frames.shape
            x = frames.view(b * n, c, h, w)
        elif frames.dim() == 6:
            b, n, k, c, h, w = frames.shape
            x = frames.view(b * n * k, c, h, w)
        else:
            return None
        x = (x + 1.0) / 2.0
        x = self._preprocess(x)
        with torch.no_grad():
            embeds = self.encoder(x)
        embeds = embeds.mean(dim=1)
        if frames.dim() == 5:
            embeds = embeds.view(b, n, -1).reshape(b, -1)
        else:
            embeds = embeds.view(b, n, k, -1).reshape(b, -1)
        return embeds

class VAEAdapter(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.available = False
        self._init_attempted = False
        self.vae = None
        self._scale_factor = 0.18215
        self._vae_paths = [
            "/mnt/sdb/lgl/EscherNet2/model/stablediffusion-v1-5/vae",
            "/mnt/sdb/lgl/EscherNet2/model/stable-diffusion-v1-5/vae"
        ]
        self._chunk_size = 64

    def _try_init(self):
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            from diffusers import AutoencoderKL
            vae_path = None
            for p in self._vae_paths:
                if os.path.isdir(p):
                    vae_path = p
                    break
            if vae_path is None:
                self.available = False
                return
            self.vae = AutoencoderKL.from_pretrained(vae_path)
            self.vae.eval()
            self.vae.to(self.device)
            self.available = True
        except Exception:
            self.available = False

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        self._try_init()
        if not self.available or self.vae is None:
            raise RuntimeError("VAE not available")
        with torch.no_grad():
            x = (x + 1.0) / 2.0
            latents = self.vae.encode(x).latent_dist.sample()
            latents = latents * self._scale_factor
        latents = latents.detach()
        return latents

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        self._try_init()
        if not self.available or self.vae is None:
            raise RuntimeError("VAE not available")
        if frames.dim() == 5:
            b, n, c, h, w = frames.shape
            total = b * n
            x = frames.view(total, c, h, w)
            x = (x + 1.0) / 2.0
            lat_list = []
            for start in range(0, total, self._chunk_size):
                end = min(start + self._chunk_size, total)
                with torch.no_grad():
                    li = self.vae.encode(x[start:end]).latent_dist.sample() * self._scale_factor
                lat_list.append(li)
            latents = torch.cat(lat_list, dim=0)
            lat_h, lat_w = latents.shape[-2], latents.shape[-1]
            latents = latents.view(b, n, 4, lat_h, lat_w)
        elif frames.dim() == 6:
            b, n, k, c, h, w = frames.shape
            total = b * n * k
            x = frames.view(total, c, h, w)
            x = (x + 1.0) / 2.0
            lat_list = []
            for start in range(0, total, self._chunk_size):
                end = min(start + self._chunk_size, total)
                with torch.no_grad():
                    li = self.vae.encode(x[start:end]).latent_dist.sample() * self._scale_factor
                lat_list.append(li)
            latents = torch.cat(lat_list, dim=0)
            lat_h, lat_w = latents.shape[-2], latents.shape[-1]
            latents = latents.view(b, n, k, 4, lat_h, lat_w)
        else:
            raise ValueError("Unsupported frames tensor shape")
        latents = latents.detach()
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        self._try_init()
        if not self.available or self.vae is None:
            raise RuntimeError("VAE not available")
        x = self.vae.decode(latents / self._scale_factor).sample
        x = x.clamp(-1.0, 1.0)
        return x

class EDM(BaseDiffusionModel):
    """
    EDM (Elucidating the Design Space of Diffusion-Based Generative Models) 扩散模型
    
    这是一个改进的扩散模型，相比传统的DDPM有以下优势：
    1. 更好的噪声调度策略
    2. 改进的损失函数设计
    3. 优化的采样算法
    4. 更稳定的训练过程
    
    支持基于历史帧和动作序列的条件图像生成。
    """
    
    @classmethod
    def from_config(
        cls,
        config: EDMConfig,
        context_length: int,
        device: str,
        model: nn.Module,
        use_vision_encoder: bool = False,
        use_latent_encoder: bool = False
    ):
        """
        从配置创建EDM实例
        
        Args:
            config: EDM配置
            context_length: 上下文长度（历史帧数量）
            device: 计算设备
            model: 底层UNet模型
            
        Returns:
            EDM实例
        """
        return cls(
            p_mean=config.p_mean,
            p_std=config.p_std,
            sigma_data=config.sigma_data,
            model=model,
            device=device,
            context_length = context_length,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            rho=config.rho,
            use_vision_encoder=use_vision_encoder,
            use_latent_encoder=use_latent_encoder
        )

    def __init__(
        self,
        p_mean: float,
        p_std: float,
        sigma_data: float,
        model: nn.Module,
        context_length: int,
        device: str,
        sigma_min = 0.002,           
        sigma_max = 80,
        rho: float = 7,
        use_vision_encoder: bool = False,
        use_latent_encoder: bool = False
    ):
        """
        初始化EDM模型
        
        Args:
            p_mean: 噪声分布的对数均值
            p_std: 噪声分布的对数标准差
            sigma_data: 数据的标准差，用于归一化
            model: 底层的去噪网络（通常是UNet）
            context_length: 历史帧的数量
            device: 计算设备
            sigma_min: 最小噪声水平
            sigma_max: 最大噪声水平
            rho: 噪声调度的形状参数
        """
        super().__init__()
        self.p_mean = p_mean      # 噪声分布参数
        self.p_std = p_std
        self.sigma_data = sigma_data  # 数据归一化参数
        self.model = model.to(device)  # 去噪网络
        self.device = device
        self.context_length = context_length  # 历史帧数量
        self.sigma_min = sigma_min    # 噪声水平范围
        self.sigma_max = sigma_max
        self.rho = rho               # 噪声调度参数
        self.action_change = ActionChange(device=device)  # 动作处理模块
        self.use_vision_encoder = use_vision_encoder
        self.cn_adapter = CNEncoderAdapter(device=device) if use_vision_encoder else None
        self.vae = VAEAdapter(device=device) if use_latent_encoder else None
        if self.vae is not None:
            self.vae._try_init()
        self.use_latent = bool(use_latent_encoder and self.vae is not None and self.vae.available)

    def _encode_frames(self, prev_frames: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_vision_encoder or self.cn_adapter is None:
            return None
        try:
            return self.cn_adapter(prev_frames)
        except Exception:
            return None

    def _denoise(self, x: torch.Tensor, sigma: torch.Tensor, prev_frames: torch.Tensor, prev_actions: torch.Tensor):
        """
        去噪函数
        
        这是EDM的核心去噪函数，使用特殊的预处理和后处理来改善训练稳定性。
        
        Args:
            x: 噪声图像
            sigma: 噪声水平
            prev_frames: 历史帧序列
            prev_actions: 历史动作序列
            
        Returns:
            去噪后的图像
        """
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        # EDM的预处理和后处理系数
        # 这些系数是根据理论分析得出的，能够改善训练稳定性
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)  # 跳跃连接系数
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()  # 输出缩放系数
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()  # 输入缩放系数
        c_noise = sigma.log() / 4  # 噪声水平编码

        if self.use_latent and self.vae and self.vae.available:
            pf = self.vae.encode_frames(prev_frames)
            if pf.dim() == 5:
                big = torch.concat([(c_in * x)[:, None, :, :, :], pf], dim=1)
            else:
                b, n, k, c, h, w = pf.shape
                pf = pf.view(b, n * k, c, h, w)
                big = torch.concat([(c_in * x)[:, None, :, :, :], pf], dim=1)
            noise_imgs = big.flatten(1,2)
        else:
            if prev_frames.dim() == 5:
                big = torch.concat([(c_in * x)[:, None, :, :, :], prev_frames], dim=1)
            else:
                b, n, k, c, h, w = prev_frames.shape
                pf = prev_frames.view(b, n * k, c, h, w)
                big = torch.concat([(c_in * x)[:, None, :, :, :], pf], dim=1)
            noise_imgs = big.flatten(1,2)
        if os.environ.get("DEBUG_STEP") == "1":
            print(f"_denoise_big {tuple(big.shape)}")
            print(f"_denoise_noise_imgs {tuple(noise_imgs.shape)}")
        
        # 处理动作序列
        self.action_change.to(self.device)
        prev_actions = self.action_change(prev_actions)
        vision_cond = self._encode_frames(prev_frames)
        if vision_cond is not None:
            prev_actions = torch.cat([prev_actions, vision_cond], dim=1)
        if os.environ.get("DEBUG_STEP") == "1":
            print(f"_denoise_actions_vec {tuple(prev_actions.shape)}")
        
        # 去噪预测
        F_x = self.model(noise_imgs, c_noise.flatten(), prev_actions)
        
        # EDM的输出组合：跳跃连接 + 缩放的网络输出
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def forward(self, imgs: torch.Tensor, prev_frames: torch.Tensor, prev_actions: torch.Tensor):
        """
        训练时的前向传播
        
        计算EDM的训练损失，使用改进的损失函数设计。
        
        Args:
            imgs: 目标图像 [B, C, H, W]
            prev_frames: 历史帧序列 [B, context_length, C, H, W]
            prev_actions: 历史动作序列 [B, context_length, ...]
            
        Returns:
            训练损失（标量）
        """
        assert prev_frames.shape[1] == prev_actions.shape[1] == self.context_length

        # 采样噪声水平
        # 使用对数正态分布采样噪声水平，这比均匀采样更有效
        rnd_normal = torch.randn([imgs.shape[0], 1, 1, 1], device=self.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        
        # EDM的损失权重，用于平衡不同噪声水平的贡献
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        if self.use_latent and self.vae and self.vae.available:
            y = self.vae.encode_image(imgs)
            n = torch.randn_like(y) * sigma
        else:
            y = imgs
            n = torch.randn_like(y) * sigma

        # 去噪预测
        D_yn = self._denoise(y + n, sigma, prev_frames, prev_actions)
        
        # 计算加权MSE损失
        return (weight * ((D_yn - y) ** 2)).mean()
        
    @torch.no_grad()
    def sample(
        self, steps: int, size: Tuple[int],
        prev_frames: torch.Tensor,
        prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        采样生成新图像
        
        使用EDM的改进采样算法生成高质量图像。
        采用二阶Heun方法进行数值积分，比一阶Euler方法更精确。
        
        Args:
            steps: 采样步数
            size: 生成图像的尺寸 (C, H, W)
            prev_frames: 历史帧序列
            prev_actions: 历史动作序列
            
        Returns:
            生成的图像
        """
        if self.use_latent and self.vae and self.vae.available:
            c, h, w = size
            h = h // 8
            w = w // 8
            x_t = torch.randn(1, 4, h, w, device=self.device)
        else:
            x_t = torch.randn(1, *size, device=self.device)

        # 设置噪声水平范围
        sigma_min = self.sigma_min
        sigma_max = self.sigma_max
        rho = self.rho

        # 时间步离散化
        # 使用特殊的噪声调度，在高噪声区域有更多步骤
        step_indices = torch.arange(steps, dtype=torch.float, device=self.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # 主采样循环
        x_next = x_t.to(torch.float) * t_steps[0]  # 初始化为最大噪声
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_hat = x_next
            t_hat = t_cur
            
            # Euler步骤（一阶近似）
            denoised = self._denoise(x_hat, t_hat, prev_frames, prev_actions).to(torch.float)
            d_cur = (x_hat - denoised) / t_hat  # 计算导数
            x_next = x_hat + (t_next - t_hat) * d_cur  # Euler更新

            # 应用二阶修正（Heun方法）
            # 这提高了采样精度，特别是在步数较少时
            if i < steps - 1:
                denoised = self._denoise(x_next, t_next, prev_frames, prev_actions).to(torch.float)
                d_prime = (x_next - denoised) / t_next  # 在新点计算导数
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)  # 二阶更新

        if self.use_latent and self.vae and self.vae.available:
            img = self.vae.decode_latents(x_next)
            return img
        return x_next
        
if __name__ == "__main__":
    """
    测试代码
    
    验证EDM模型的基本功能
    """
    # 测试参数
    size = (64, 64)          # 图像尺寸
    input_channels = 3       # RGB通道
    context_length = 4       # 历史帧数量
    actions_count = 5        # 动作类别数
    batch_size = 3          # 批次大小

    from blocks import UNet

    # 创建UNet网络
    unet = UNet((input_channels) * (context_length + 1), 3, None, actions_count, context_length)
    
    # 创建EDM模型
    ddpm = EDM(
        p_mean=-1.2,           # 噪声分布参数
        p_std=1.2,
        sigma_data=0.5,        # 数据标准差
        model=unet,
        context_length=context_length,
        device="cpu"
    )

    # 准备测试数据
    img = torch.randn((batch_size, input_channels, *size))  # 目标图像
    prev_frames = torch.randn((batch_size, context_length, input_channels, *size))  # 历史帧
    prev_actions = torch.randint(low=0, high=actions_count, size=(batch_size, context_length))  # 历史动作
    
    # 测试前向传播（训练损失计算）
    ddpm.forward(img, prev_frames, prev_actions)
