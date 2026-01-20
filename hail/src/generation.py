"""
生成配置模块

定义扩散模型生成过程中使用的配置参数，包括图像尺寸、通道数、序列长度等。
这些配置参数用于初始化模型架构和控制生成过程。

作者：Snake Diffusion项目组
"""

from dataclasses import dataclass

@dataclass
class GenerationConfig:
    """
    生成配置类
    
    包含扩散模型生成过程中需要的所有配置参数
    
    Attributes:
        image_size: 图像尺寸（假设为正方形图像）
        input_channels: 输入图像通道数（通常为3，RGB）
        output_channels: 输出图像通道数（通常为3，RGB）
        context_length: 上下文长度（历史帧数量）
        actions_count: 动作类别数量
    """
    image_size: int
    input_channels: int
    output_channels: int
    context_length: int
    actions_count: int
    use_latent: bool = False
    latent_channels: int = 4
    neighborhood: int = 1

    @property
    def unet_input_channels(self) -> int:
        """
        计算UNet模型的输入通道数
        
        UNet需要同时处理当前帧和历史帧，因此输入通道数为：
        输入通道数 × (历史帧数 + 当前帧数)
        
        Returns:
            UNet模型的总输入通道数
        """
        base = self.latent_channels if self.use_latent else self.input_channels
        return base * (1 + self.context_length * max(1, int(self.neighborhood)))
