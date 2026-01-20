"""
雷达回波外推扩散模型训练脚本

这个脚本专门用于训练雷达回波外推任务的扩散模型，是原Snake游戏扩散模型的改进版本。
主要针对气象雷达数据进行短期降水预报和强对流天气预警。

核心功能：
1. 雷达回波数据加载：使用HailDataset处理多区域雷达反射率图像
2. 气象参数融合：整合风向、风速等气象要素提升预测精度
3. 时序建模：基于历史雷达帧预测未来降水分布
4. 多模型支持：支持EDM和DDPM两种扩散模型架构
5. 验证图像生成：实时生成预测结果用于模型效果评估

技术特点：
- 多尺度数据融合：64x64雷达图像 + 16x16气象参数
- 时序条件生成：基于历史4帧预测下一帧
- 分布式训练：支持GPU加速和多卡训练
- 实时验证：训练过程中生成预测样本进行效果评估

应用场景：
- 短期降水预报（0-2小时）
- 强对流天气预警
- 极端天气事件预测
- 雷达回波外推

作者：雷达回波外推项目组
基于：Snake Diffusion项目
"""

from typing import List
import os
import random
# 设置CUDA设备，使用第0号GPU进行雷达回波外推模型训练
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import click
import yaml
import pickle
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 导入训练相关模块
from training_utils.train_loop import train_loop, TrainingConfig
from generation import GenerationConfig
from models.gen.blocks import BaseDiffusionModel, UNet, UNetConfig
from models.gen.edm import EDM, EDMConfig
from models.gen.ddpm import DDPM, DDPMConfig
from utils.utils import EasyDict, instantiate_from_config
from data.data import SequencesDataset
from data.data import HailDataset  # 雷达回波数据集
from models.gen.edm import ActionChange  # 气象参数处理模块

def _save_sample_imgs(
    frames_real: torch.Tensor,
    frames_gen: List[torch.Tensor],
    path: str
):
    """
    保存雷达回波预测结果对比图像
    
    该函数用于生成训练过程中的验证图像，将真实雷达回波序列与模型生成的预测序列
    进行可视化对比，帮助评估模型的预测效果和训练进度。
    
    Args:
        frames_real: 真实雷达回波序列 [T, C, H, W]
                    包含连续时刻的真实雷达反射率图像
        frames_gen: 模型生成的预测序列列表 List[[T, C, H, W]]
                   每个元素是不同采样步数下的预测结果
                   通常包含[10步采样, 5步采样, 2步采样]的结果
        path: 保存路径，如"val_images/epoch_10.png"
        
    图像布局：
    - 第1行：真实雷达回波序列（Ground Truth）
    - 第2行：10步采样预测结果
    - 第3行：5步采样预测结果  
    - 第4行：2步采样预测结果
    
    每列代表一个时间步，从左到右显示时序演变过程
    """
    def get_np_img(tensor: torch.Tensor) -> np.ndarray:
        """
        将PyTorch张量转换为可显示的numpy图像
        
        处理流程：
        1. 反归一化：从[-1,1]范围恢复到[0,255]
        2. 数据类型转换：float -> uint8
        3. 维度调整：CHW -> HWC（适配matplotlib显示格式）
        
        Args:
            tensor: 归一化的图像张量 [C, H, W]，值域[-1, 1]
            
        Returns:
            可显示的图像数组 [H, W, C]，值域[0, 255]
        """
        return (tensor * 127.5 + 127.5).long().clip(0,255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

    # 设置图像布局参数
    height_row = 5  # 每行高度（英寸）
    col_width = 5   # 每列宽度（英寸）
    cols = len(frames_real)  # 列数：时间步数量
    rows = 1 + len(frames_gen)  # 行数：1行真实 + N行生成
    
    # 创建子图网格
    fig, axes = plt.subplots(rows, cols, figsize=(col_width * cols, height_row * rows))
    
    # 填充每个子图
    for row in range(rows):
        # 第0行显示真实序列，其他行显示对应的生成序列
        frames = frames_real if row == 0 else frames_gen[row - 1]
        for i in range(len(frames_real)):
            axes[row, i].imshow(get_np_img(frames[i]))
            # 移除坐标轴，使图像更清晰
            axes[row, i].axis('off')
            
    # 调整子图间距，使图像紧密排列
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # 保存图像并释放内存
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

def _generate_and_save_sample_imgs(
    model: BaseDiffusionModel,
    dataset: HailDataset,
    epoch: int,
    device: str,
    context_length: int,
    length_session = 20
):
    """
    生成并保存雷达回波预测验证图像
    
    该函数在训练过程中定期调用，用于生成模型预测的雷达回波序列，
    并与真实序列进行对比，以评估模型的预测能力和训练效果。
    
    核心流程：
    1. 从数据集中随机选择一个连续的雷达回波序列
    2. 使用不同采样步数（2步、5步、10步）生成预测序列
    3. 将真实序列与预测序列保存为对比图像
    
    Args:
        model: 训练中的扩散模型
        dataset: 雷达回波数据集
        epoch: 当前训练轮次
        device: 计算设备（cuda/cpu）
        context_length: 历史帧长度（用于预测的历史帧数量）
        length_session: 生成序列的长度（默认20帧）
        
    生成策略：
    - 10步采样：高质量但速度较慢，用于评估模型最佳性能
    - 5步采样：质量与速度的平衡，实际应用的推荐设置
    - 2步采样：快速生成，用于实时预警场景
    
    输出文件：val_images/{epoch}.png
    """
    # 确保序列长度不超过数据集大小
    if len(dataset) - 1 < length_session:
        length_session = len(dataset) - 1
    
    # 随机选择起始索引，确保有足够的连续帧
    index = random.randint(0, len(dataset) - 1 - length_session)

    # 获取初始数据：目标图像、历史帧、气象参数
    img, last_imgs, actions = dataset[index]

    # 将数据移动到计算设备
    img = img.to(device)
    last_imgs = last_imgs.to(device)  # 历史雷达帧
    actions = actions.to(device)      # 对应的气象参数

    # 初始化序列存储：真实序列和不同采样步数的生成序列
    real_imgs = last_imgs.clone()     # 真实雷达回波序列
    gen_2_imgs = last_imgs.clone()    # 2步采样生成序列
    gen_10_imgs = last_imgs.clone()   # 10步采样生成序列
    gen_5_imgs = last_imgs.clone()    # 5步采样生成序列
    
    # 逐步生成预测序列
    for j in range(1, length_session):
        # 获取下一时刻的真实数据
        img, last_imgs, actions = dataset[index + j]
        img = img.to(device)
        last_imgs = last_imgs.to(device)
        actions = actions.to(device)
        
        # 添加真实图像到序列
        real_imgs = torch.concat([real_imgs, img[None, :, :, :]], dim=0)
        
        # 使用10步采样生成预测图像
        # 基于最近context_length帧历史和当前气象参数进行预测
        gen_img = model.sample(10, img.shape, gen_10_imgs[-context_length:].unsqueeze(0), actions.unsqueeze(0))[0]
        gen_10_imgs = torch.concat([gen_10_imgs, gen_img[None, :, :, :]], dim=0)
        
        # 使用2步采样生成预测图像（快速预测）
        gen_img = model.sample(2, img.shape, gen_2_imgs[-context_length:].unsqueeze(0), actions.unsqueeze(0))[0]
        gen_2_imgs = torch.concat([gen_2_imgs, gen_img[None, :, :, :]], dim=0)
        
        # 使用5步采样生成预测图像（平衡质量与速度）
        gen_img = model.sample(5, img.shape, gen_5_imgs[-context_length:].unsqueeze(0), actions.unsqueeze(0))[0]
        gen_5_imgs = torch.concat([gen_5_imgs, gen_img[None, :, :, :]], dim=0)

    # 保存对比图像：真实序列 vs 不同采样步数的预测序列
    _save_sample_imgs(real_imgs, [gen_10_imgs, gen_5_imgs, gen_2_imgs], f"val_images/{epoch}.png")

@click.command()
@click.option('--config', help='训练配置文件路径（YAML格式）', metavar='YAML', type=str, required=True, default="config/Diffusion.yaml")
@click.option('--model-type', type=click.Choice(['ddpm', 'edm'], case_sensitive=False), default='edm', help='扩散模型类型：edm（推荐）或ddpm')
@click.option('--output-prefix', help='模型输出路径前缀，将添加轮次后缀', type=str, required=True)

@click.option('--dataset', help='雷达回波数据集根目录路径', type=str, required=False)
@click.option('--output-loader', help='保存数据加载器的路径（用于缓存数据集分割）', type=str, required=False)
@click.option('--loader', help='预处理的数据加载器路径（跳过数据集处理）', type=str, required=False)

@click.option('--gen-val-images', help='是否生成验证图像（用于监控训练效果）', is_flag=True, required=False, default=False)

@click.option('--last-checkpoint', help='恢复训练的检查点路径', type=str, required=False)
@click.option('--image-size', help='覆盖配置中的图像尺寸', type=int, required=False)
def main(**kwargs):
    """
    雷达回波外推扩散模型训练主函数
    
    该函数是整个训练流程的入口点，负责：
    1. 解析命令行参数和配置文件
    2. 初始化扩散模型（EDM或DDPM）
    3. 加载和预处理雷达回波数据集
    4. 配置训练和验证数据加载器
    5. 启动训练循环
    
    使用示例：
    python train2.py --config config/radar.yaml --model-type edm --output-prefix models/radar_edm --dataset /data/radar --gen-val-images
    
    参数说明：
    - config: 包含模型架构、训练超参数等配置的YAML文件
    - model-type: 选择EDM（推荐，更好的生成质量）或DDPM（经典扩散模型）
    - output-prefix: 模型检查点保存路径前缀
    - dataset: 雷达数据根目录，包含images/和final/子目录
    - gen-val-images: 启用后在训练过程中生成预测样本用于效果评估
    """
    # 解析命令行参数
    options = EasyDict(kwargs)
    
    # 加载训练配置文件
    with open(options.config, 'r') as f:
        config = EasyDict(**yaml.safe_load(f))
    
    # 解析配置参数
    training_config = TrainingConfig(**config.training)      # 训练超参数配置
    generation_config = GenerationConfig(**config.generation)  # 生成模型配置

    # 定义图像预处理变换
    # 将PIL图像转换为张量并归一化到[-1, 1]范围（适配扩散模型）
    _img_size = int(options.image_size) if options.get('image_size') else int(generation_config.image_size)
    transform_to_tensor = transforms.Compose([
        transforms.Resize((_img_size, _img_size)),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    ])

    # 自动检测计算设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Mac OS支持（Apple Silicon GPU加速）
    if torch.backends.mps.is_available():
        device = "mps"

    # 初始化扩散模型
    model: BaseDiffusionModel
    if options.model_type == "edm":
        # EDM (Elucidating the Design Space of Diffusion-Based Generative Models)
        # 更先进的扩散模型，具有更好的生成质量和训练稳定性
        config = EDMConfig(**instantiate_from_config(config.edm))
        model = EDM.from_config(
            config=config,
            context_length=generation_config.context_length,  # 历史帧数量
            device=device,
            model=UNet.from_config(
                config=config.unet,
                # 输入通道：历史帧 + 当前帧的通道数
                in_channels=generation_config.unet_input_channels,
                out_channels=generation_config.output_channels,  # 输出通道（RGB）
                actions_count=generation_config.actions_count,   # 气象参数维度
                seq_length=generation_config.context_length     # 序列长度
            )
        )
    elif options.model_type == "ddpm":
        # DDPM (Denoising Diffusion Probabilistic Models)
        # 经典扩散模型，训练稳定但生成质量略低于EDM
        config = DDPMConfig(**instantiate_from_config(config.ddpm))
        model = DDPM.from_config(
            config=config,
            context_length=generation_config.context_length,
            device=device,
            model=UNet.from_config(
                config=config.unet,
                in_channels=generation_config.unet_input_channels,
                out_channels=generation_config.output_channels,
                actions_count=generation_config.actions_count,
                seq_length=generation_config.context_length,
                T=config.T  # DDPM特有的扩散步数参数
            )
        )

    # 数据集加载和预处理
    if options.dataset:
        # 从原始数据目录创建雷达回波数据集
        dataset = HailDataset(
            images_dir=os.path.join(options.dataset),           # 雷达图像目录
            actions_path=os.path.join(options.dataset, "final"), # 气象参数目录
            seq_length=generation_config.context_length,        # 历史序列长度
            transform=transform_to_tensor                        # 图像预处理变换
        )

        # 数据集分割：80%训练，20%验证
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        valid_size = total_size - train_size

        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, valid_size]
        )

        # 可选：保存数据集分割结果，避免重复处理
        if options.output_loader:
            with open(options.output_loader, 'wb') as f:
                pickle.dump({
                    "train": train_dataset,
                    "val": val_dataset,
                    "all": dataset
                }, f)
    elif options.loader:
        # 从预处理的数据加载器文件加载数据集
        with open(options.loader, 'rb') as f:
            checkpoint = pickle.load(f)
            train_dataset = checkpoint['train']
            train_dataset.transform = transform_to_tensor
            val_dataset = checkpoint['val']
            val_dataset.transform = transform_to_tensor
            dataset = checkpoint['all']
            dataset.transform = transform_to_tensor

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,    # 批次大小
        shuffle=True,                             # 训练时打乱数据
        num_workers=training_config.num_workers   # 多进程加载
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers   # 验证时不打乱数据
    )

    # 定义验证图像生成回调函数
    def gen_val_images(epoch: int):
        """在指定轮次生成验证图像"""
        _generate_and_save_sample_imgs(model, dataset, epoch, device, generation_config.context_length)

    # 启动训练循环
    print(f"开始训练 {options.model_type.upper()} 雷达回波外推模型...")
    train_loop(
        model=model,                                              # 扩散模型
        device=device,                                            # 计算设备
        config=training_config,                                   # 训练配置
        train_dataloader=train_dataloader,                        # 训练数据
        val_dataloader=val_dataloader,                            # 验证数据
        output_path_prefix=options.output_prefix,                 # 模型保存路径
        existing_model_path=options["last_checkpoint"],           # 恢复检查点
        gen_imgs=gen_val_images if options.gen_val_images else None,  # 验证图像生成
    )

if __name__ == "__main__":
    # 程序入口点：启动雷达回波外推扩散模型训练
    main()
