"""
扩散模型架构模块

这个模块实现了扩散模型的核心架构组件，包括：
1. 基础扩散模型抽象类
2. 位置编码（时间步嵌入）
3. UNet架构的各种构建块
4. 注意力机制
5. 残差块和上下采样块
6. 完整的UNet模型

主要用于图像生成任务，特别是基于历史帧和动作序列的条件生成。

作者：Snake Diffusion项目组
"""

from typing import List, Optional, Tuple
from functools import partial
import abc
from abc import ABC
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os

class BaseDiffusionModel(nn.Module, ABC):
    """
    扩散模型基类
    
    定义了所有扩散模型必须实现的接口，主要是采样方法。
    这是一个抽象基类，具体的扩散模型（如EDM、DDPM）需要继承并实现。
    """
    
    @torch.no_grad()
    @abc.abstractmethod
    def sample(self, steps: int, size: Tuple[int],
        prev_frames: torch.Tensor,
        prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        抽象采样方法
        
        Args:
            steps: 扩散步数
            size: 生成图像的尺寸
            prev_frames: 历史帧序列
            prev_actions: 历史动作序列
            
        Returns:
            生成的图像张量
        """
        pass
    
class PositionalEmbedding(nn.Module):
    """
    位置编码模块（用于离散时间步）
    
    使用正弦和余弦函数为时间步生成位置编码，类似于Transformer中的位置编码。
    这种编码方式能够让模型理解不同时间步之间的相对关系。
    """
    
    def __init__(self, T: int, output_dim: int) -> None:
        """
        初始化位置编码
        
        Args:
            T: 最大时间步数
            output_dim: 输出维度
        """
        super().__init__()
        self.output_dim = output_dim
        
        # 创建位置索引
        position = torch.arange(T+1).unsqueeze(1)
        
        # 计算频率项，用于生成不同频率的正弦余弦波
        div_term = torch.exp(torch.arange(0, output_dim, 2) * (-math.log(10000.0) / output_dim))
        
        # 初始化位置编码矩阵
        pe = torch.zeros(T+1, output_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦
        
        # 注册为缓冲区，不参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """
        前向传播
        
        Args:
            x: 时间步索引张量
            
        Returns:
            对应的位置编码
        """
        return self.pe[x].reshape(x.shape[0], self.output_dim)
    
class FloatPositionalEmbedding(nn.Module):
    """
    浮点位置编码模块（用于连续时间步）
    
    与PositionalEmbedding类似，但支持连续的浮点时间步，
    适用于连续时间的扩散模型。
    """
    
    def __init__(self, output_dim: int, max_positions=10000) -> None:
        """
        初始化浮点位置编码
        
        Args:
            output_dim: 输出维度
            max_positions: 最大位置数，用于频率计算
        """
        super().__init__()
        self.output_dim = output_dim
        self.max_positions = max_positions

    def forward(self, x: torch.Tensor):
        """
        前向传播
        
        Args:
            x: 连续时间步张量
            
        Returns:
            对应的位置编码
        """
        # 计算频率
        freqs = torch.arange(start=0, end=self.output_dim//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.output_dim // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        
        # 计算位置编码
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    
# 定义常用的卷积层
Conv1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)  # 1x1卷积，用于通道变换
Conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)  # 3x3卷积，保持空间尺寸

# GroupNorm相关常量
GROUP_SIZE = 32  # 组归一化的组大小
GN_EPS = 1e-5   # 组归一化的epsilon值

class GroupNorm(nn.Module):
    """
    组归一化模块
    
    GroupNorm是BatchNorm的替代方案，在扩散模型中表现更好。
    它将通道分组进行归一化，不依赖于批次大小。
    """
    
    def __init__(self, in_channels: int) -> None:
        """
        初始化组归一化
        
        Args:
            in_channels: 输入通道数
        """
        super().__init__()
        # 计算组数，确保至少有1组
        num_groups = max(1, in_channels // GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.norm(x)

class MultiheadAttention(nn.Module):
    """
    多头注意力模块
    
    实现了空间自注意力机制，让模型能够关注图像中的长距离依赖关系。
    这对于生成高质量的图像非常重要。
    """
    
    def __init__(self, in_channels: int, head_dim: int = 8) -> None:
        """
        初始化多头注意力
        
        Args:
            in_channels: 输入通道数
            head_dim: 每个注意力头的维度
        """
        super().__init__()
        # 计算注意力头数
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        
        # 归一化层
        self.norm = GroupNorm(in_channels)
        
        # QKV投影层，一次性计算查询、键、值
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        
        # 输出投影层
        self.out_proj = Conv1x1(in_channels, in_channels)
        
        # 初始化输出投影为零，实现残差连接的稳定训练
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [N, C, H, W]
            
        Returns:
            注意力处理后的特征图
        """
        n, c, h, w = x.shape
        
        # 归一化
        x = self.norm(x)
        
        # 计算QKV
        qkv = self.qkv_proj(x)
        qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
        q, k, v = [x for x in qkv.chunk(3, dim=1)]
        
        # 计算注意力权重
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        
        # 应用注意力权重
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        
        # 残差连接
        return x + self.out_proj(y)

class NormBlock(nn.Module):
    """
    条件归一化块
    
    结合了组归一化和条件信息（如时间步嵌入），
    让模型能够根据条件调整归一化参数。
    """
    
    def __init__(self, in_channels, cond_channels):
        """
        初始化条件归一化块
        
        Args:
            in_channels: 输入通道数
            cond_channels: 条件信息通道数
        """
        super().__init__()
        self.norm = nn.GroupNorm(max(in_channels // GROUP_SIZE, 1), in_channels)
        self.ln = nn.Linear(cond_channels, in_channels)

    def forward(self, x, cond):
        """
        前向传播
        
        Args:
            x: 输入特征图
            cond: 条件信息（如时间步嵌入）
            
        Returns:
            条件归一化后的特征图
        """
        return self.norm(x) + self.ln(cond)[:, :, None, None]

class ResnetBlock(nn.Module):
    """
    残差块
    
    扩散模型的核心构建块，包含两个卷积层、条件归一化和可选的注意力机制。
    残差连接有助于训练深层网络和梯度流动。
    """
    
    def __init__(self, in_channels, out_channels, cond_channels, has_attn=False):
        """
        初始化残差块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            cond_channels: 条件信息通道数
            has_attn: 是否包含注意力机制
        """
        super().__init__()
        
        # 通道匹配投影（如果输入输出通道数不同）
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        
        # 第一个归一化和卷积
        self.norm_1 = NormBlock(out_channels, cond_channels)
        self.conv_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        )
        
        # 第二个归一化和卷积
        self.norm_2 = NormBlock(out_channels, cond_channels)
        self.conv_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )
        
        # 可选的注意力机制
        self.attn = nn.Identity() if not has_attn else MultiheadAttention(out_channels)

    def forward(self, x, cond):
        """
        前向传播
        
        Args:
            x: 输入特征图
            cond: 条件信息
            
        Returns:
            处理后的特征图
        """
        h = self.proj(x)  # 通道匹配
        x = self.conv_1(self.norm_1(h, cond))  # 第一个卷积块
        x = self.conv_2(self.norm_2(x, cond))  # 第二个卷积块
        return self.attn(h + x)  # 残差连接 + 可选注意力
    
class DownBlock(nn.Module):
    """
    下采样块
    
    使用步长为2的卷积进行下采样，将特征图尺寸减半。
    这是UNet编码器部分的关键组件。
    """
    
    def __init__(self, in_channels: int):
        """
        初始化下采样块
        
        Args:
            in_channels: 输入通道数
        """
        super().__init__()
        self.pool = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        """前向传播，将特征图尺寸减半"""
        return self.pool(x)
    
class UpBlock(nn.Module):
    """
    上采样块
    
    使用最近邻插值进行上采样，将特征图尺寸加倍。
    这是UNet解码器部分的关键组件。
    """
    
    def __init__(self, in_channels: int) -> None:
        """
        初始化上采样块
        
        Args:
            in_channels: 输入通道数
        """
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，将特征图尺寸加倍
        
        Args:
            x: 输入特征图
            
        Returns:
            上采样后的特征图
        """
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")  # 最近邻插值上采样
        return self.conv(x)  # 卷积平滑

class ResnetsBlock(nn.Module):
    """
    残差块序列
    
    将多个残差块串联起来，形成更深的网络结构。
    这样可以增加模型的表达能力。
    """
    
    def __init__(self, in_channels_list: List[int], out_channels_list: List[int], cond_channels: int, has_attn=False):
        """
        初始化残差块序列
        
        Args:
            in_channels_list: 每个残差块的输入通道数列表
            out_channels_list: 每个残差块的输出通道数列表
            cond_channels: 条件信息通道数
            has_attn: 是否包含注意力机制
        """
        super().__init__()
        assert len(in_channels_list) == len(out_channels_list)
        self.models = nn.ModuleList([
            ResnetBlock(in_ch, out_ch, cond_channels, has_attn) for in_ch, out_ch in zip(in_channels_list, out_channels_list)
        ])
    
    def forward(self, x, cond):
        """
        前向传播，依次通过所有残差块
        
        Args:
            x: 输入特征图
            cond: 条件信息
            
        Returns:
            处理后的特征图
        """
        for module in self.models:
            x = module(x, cond)
        return x
    
@dataclass
class UNetConfig:
    """
    UNet配置类
    
    包含UNet架构的所有配置参数
    
    Attributes:
        steps: 每个分辨率级别的残差块数量
        channels: 每个分辨率级别的通道数
        cond_channels: 条件信息的通道数
        attn_step_indexes: 哪些级别使用注意力机制
    """
    steps: List[int]
    channels: List[int]
    cond_channels: int
    attn_step_indexes: List[bool]

class UNet(nn.Module):
    """
    UNet模型
    
    这是扩散模型的核心网络架构，采用编码器-解码器结构：
    1. 编码器：逐步下采样，提取多尺度特征
    2. 瓶颈层：最深层的特征处理
    3. 解码器：逐步上采样，结合跳跃连接重建图像
    
    支持条件生成，可以基于时间步和历史动作进行图像生成。
    """
    
    @classmethod
    def from_config(
        cls,
        config: UNetConfig,
        in_channels: int,
        out_channels: int,
        actions_count: int,
        seq_length: int,
        T: Optional[int] = None
    ):
        """
        从配置创建UNet实例
        
        Args:
            config: UNet配置
            in_channels: 输入通道数
            out_channels: 输出通道数
            actions_count: 动作类别数
            seq_length: 序列长度
            T: 时间步数（None表示连续时间）
            
        Returns:
            UNet实例
        """
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            T=T,
            actions_count=actions_count,
            seq_length=seq_length,
            steps=config.steps,
            channels=config.channels,
            cond_channels=config.cond_channels,
            attn_step_indexes=config.attn_step_indexes
        )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        T: Optional[int],
        actions_count: int,
        seq_length: int,
        steps=(2, 2, 2, 2),
        channels = (64, 64, 64, 64),
        cond_channels = 256,
        attn_step_indexes = [False, False, False, False]
    ):
        """
        初始化UNet模型
        
        Args:
            in_channels: 输入通道数（通常是RGB通道数乘以序列长度）
            out_channels: 输出通道数（通常是RGB通道数）
            T: 扩散时间步数（None表示连续时间）
            actions_count: 动作类别数
            seq_length: 历史序列长度
            steps: 每个分辨率级别的残差块数量
            channels: 每个分辨率级别的通道数
            cond_channels: 条件嵌入的维度
            attn_step_indexes: 哪些级别使用注意力机制
        """
        super().__init__()
        assert len(steps) == len(channels) == len(attn_step_indexes)
        
        # 时间步嵌入：将时间步转换为高维向量
        self.time_embedding = PositionalEmbedding(T=T, output_dim=cond_channels) if T is not None else FloatPositionalEmbedding(output_dim=cond_channels)
        
        # 动作嵌入：将动作序列转换为向量（当前实现较简单）
        self.actions_embedding = nn.Sequential(
            nn.Embedding(actions_count, cond_channels // seq_length),
            nn.Flatten()
        )
        
        # 条件嵌入处理：融合时间和动作信息
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.ReLU(),
            nn.Linear(cond_channels, cond_channels),
        )

        # 动作向量线性投影（在首次前向时根据实际维度动态初始化）
        self.actions_linear: Optional[nn.Linear] = None

        # 第一层卷积：将输入映射到第一个特征通道
        self.first_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        
        # 构建编码器和解码器
        down_res_blocks = []  # 编码器残差块
        self.downsample_blocks = nn.ModuleList()  # 下采样块
        self.upsample_blocks = nn.ModuleList()   # 上采样块
        up_res_blocks = []    # 解码器残差块
        
        for (index, step) in enumerate(steps):
            # 编码器残差块配置
            in_ch = step * [channels[index]]
            out_ch = in_ch.copy()
            out_ch[-1] = channels[index + 1] if index < len(steps) - 1 else channels[index]
            
            down_res_blocks.append(ResnetsBlock(
                in_channels_list=in_ch,
                out_channels_list=out_ch,
                cond_channels=cond_channels,
                has_attn=attn_step_indexes[index]
            ))
            
            # 下采样和上采样块
            self.downsample_blocks.append(DownBlock(in_ch[0]))
            self.upsample_blocks.append(UpBlock(out_ch[-1]))
            
            # 解码器残差块配置（包含跳跃连接）
            in_ch = step * [channels[index]]
            out_ch = in_ch.copy()
            in_ch[0] = 2 * (channels[index + 1] if index < len(steps) - 1 else channels[index])  # 跳跃连接使通道数翻倍
            
            up_res_blocks.append(ResnetsBlock(
                in_channels_list=in_ch,
                out_channels_list=out_ch,
                cond_channels=cond_channels,
                has_attn=attn_step_indexes[index]
            ))
            
        self.downres_blocks = nn.ModuleList(down_res_blocks)
        self.upres_blocks = nn.ModuleList(reversed(up_res_blocks))  # 解码器顺序相反
        
        # 瓶颈层：最深层的特征处理，通常包含注意力
        self.backbone = ResnetsBlock(
            [channels[-1]] * 2,
            [channels[-1]] * 2,
            cond_channels=cond_channels,
            has_attn=True
        )
        
        # 输出层：将特征映射回图像空间
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[0], out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, prev_actions: torch.Tensor):
        """
        UNet前向传播
        
        Args:
            x: 输入图像张量 [B, C, H, W]
            t: 时间步张量 [B] 或 [B, 1]
            prev_actions: 历史动作序列 [B, seq_len] 或 [B, action_dim]
            
        Returns:
            预测的噪声或图像 [B, out_channels, H, W]
        """
        assert x.shape[0] == prev_actions.shape[0]
        
        # 生成条件嵌入
        time_emb = self.time_embedding(t)
        # 直接使用来自EDM.ActionChange的动作向量；若维度不匹配，投影到cond_channels
        actions_emb = prev_actions
        if actions_emb.dim() == 1:
            actions_emb = actions_emb.unsqueeze(0)
        if actions_emb.shape[1] != time_emb.shape[1]:
            # 动态初始化线性层以匹配cond_channels
            if self.actions_linear is None:
                self.actions_linear = nn.Linear(actions_emb.shape[1], time_emb.shape[1]).to(actions_emb.device)
            actions_emb = self.actions_linear(actions_emb)
        cond = self.cond_embedding(time_emb + actions_emb)
        if os.environ.get("DEBUG_STEP") == "1":
            print(f"unet_time_emb {tuple(time_emb.shape)}")
            print(f"unet_actions_emb {tuple(actions_emb.shape)}")
            print(f"unet_cond {tuple(cond.shape)}")

        # 第一层卷积
        x = self.first_conv(x)
        if os.environ.get("DEBUG_STEP") == "1":
            print(f"unet_x_first_conv {tuple(x.shape)}")
        
        # 编码器路径：逐步下采样并保存跳跃连接
        hx = []  # 保存每层的特征用于跳跃连接
        for index, downres_block in enumerate(self.downres_blocks):
            x = downres_block(x, cond)  # 残差处理
            hx.append(x)                # 保存特征
            x = self.downsample_blocks[index](x)  # 下采样
                
        # 瓶颈层处理
        x = self.backbone(x, cond)

        # 解码器路径：逐步上采样并融合跳跃连接
        for index, up_block in enumerate(self.upres_blocks):
            x = self.upsample_blocks[len(self.upres_blocks) - index - 1](x)  # 上采样
            # 融合跳跃连接：将当前特征与对应编码器层的特征拼接
            x = up_block(torch.cat([x, hx[len(self.upres_blocks) - index - 1]], 1), cond)
            
        # 输出层
        x = self.out(x)
        return x
    
if __name__ == "__main__":
    """
    测试代码
    
    验证各个模块的功能是否正常
    """
    # 测试下采样块
    DownBlock(4).forward(torch.rand(4,4,64,64)).size(-1) == 32
    DownBlock(4).forward(torch.rand(4,4,32,32)).size(-1) == 16

    # 测试完整UNet
    size = (64, 64)          # 图像尺寸
    input_channels = 3       # RGB通道
    context_length = 4       # 历史帧数量
    actions_count = 5        # 动作类别数
    T = 1000                # 扩散步数
    batch_size = 2          # 批次大小
    
    # 创建UNet实例
    unet = UNet(
        (input_channels) * (context_length + 1),  # 输入通道：当前帧+历史帧
        3,                                        # 输出通道：RGB
        None,                                     # 连续时间
        actions_count,
        context_length
    )
    
    # 准备测试数据
    img = torch.randn((batch_size, input_channels, *size))  # 当前帧
    prev_frames = torch.randn((batch_size, context_length, input_channels, *size))  # 历史帧
    frames = torch.concat([img[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)  # 拼接所有帧

    prev_actions = torch.randint(low=0, high=actions_count, size=(batch_size, context_length))  # 历史动作
    t = torch.randn((batch_size,))  # 时间步
    
    # 前向传播测试
    unet.forward(frames, t.unsqueeze(1), prev_actions)
