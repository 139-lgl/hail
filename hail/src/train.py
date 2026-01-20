"""
Snake扩散模型训练脚本

这个脚本是整个项目的核心训练模块，负责训练扩散模型来生成Snake游戏序列。
支持EDM和DDPM两种扩散模型架构，可以从Q学习智能体收集的数据中学习游戏行为。

主要功能：
1. 加载和预处理训练数据
2. 初始化扩散模型（EDM或DDPM）
3. 执行训练循环
4. 生成验证图像
5. 保存模型检查点

作者：Snake Diffusion项目组
"""

from typing import List
import os
import random
# 设置CUDA设备，指定使用第3号GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import click
import yaml
import pickle
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, Subset
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
from data.data import HailDataset
from models.gen.edm import ActionChange, VAEAdapter
from data.hourly_dataset import HourlyHailDataset
from data.data import TiledHailDataset

def _save_sample_imgs(
    frames_real: torch.Tensor,
    frames_gen: List[torch.Tensor],
    path: str
):
    """
    保存样本图像的可视化对比图
    
    将真实图像序列与生成的图像序列进行对比展示，用于评估模型性能
    
    Args:
        frames_real: 真实的图像序列张量 [seq_len, C, H, W]
        frames_gen: 生成的图像序列列表，每个元素是一个张量 [seq_len, C, H, W]
        path: 保存图像的路径
    """
    def get_np_img(tensor: torch.Tensor) -> np.ndarray:
        """将张量转换为可显示的numpy数组"""
        # 反归一化：从[-1,1]范围转换到[0,255]范围
        return (tensor * 127.5 + 127.5).long().clip(0,255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

    # 设置图像网格的尺寸
    height_row = 5  # 每行的高度
    col_width = 5   # 每列的宽度
    cols = len(frames_real)  # 列数等于序列长度
    rows = 1 + len(frames_gen)  # 行数：1行真实图像 + N行生成图像
    
    # 创建子图网格
    fig, axes = plt.subplots(rows, cols, figsize=(col_width * cols, height_row * rows))
    
    # 填充每个子图
    for row in range(rows):
        # 第0行显示真实图像，其他行显示生成图像
        frames = frames_real if row == 0 else frames_gen[row - 1]
        for i in range(len(frames_real)):
            axes[row, i].imshow(get_np_img(frames[i]))
            
    # 调整子图间距
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # 保存图像
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def _generate_and_save_sample_imgs(
    model: BaseDiffusionModel,
    dataset: HailDataset,
    epoch: int,
    device: str,
    context_length: int,
    length_session = 20,
    val_images_dir: str = "val_images_1"
):
    """
    生成并保存验证图像
    
    使用训练好的模型生成图像序列，并与真实序列进行对比，
    用不同的采样步数（2, 5, 10）来评估模型在不同质量-速度权衡下的表现
    
    Args:
        model: 训练好的扩散模型
        dataset: 数据集
        epoch: 当前训练轮次
        device: 计算设备（cuda/cpu/mps）
        context_length: 上下文长度（历史帧数）
        length_session: 生成序列的长度
    """
    # 确保序列长度不超过数据集大小
    if len(dataset) - 1 < length_session:
        length_session = len(dataset) - 1
    
    # 随机选择一个起始索引
    index = random.randint(0, len(dataset) - 1 - length_session)

    # 获取初始数据
    img, last_imgs, actions = dataset[index]

    # 将数据移动到指定设备
    img = img.to(device)
    last_imgs = last_imgs.to(device)
    actions = actions.to(device)

    # 初始化不同采样步数的图像序列
    if last_imgs.dim() == 4:
        real_imgs = last_imgs.clone()
        gen_2_imgs = last_imgs.clone()
        gen_10_imgs = last_imgs.clone()
        gen_5_imgs = last_imgs.clone()
    else:
        real_imgs = last_imgs[:, 0].clone()
        gen_2_imgs = last_imgs[:, 0].clone()
        gen_10_imgs = last_imgs[:, 0].clone()
        gen_5_imgs = last_imgs[:, 0].clone()
    
    # 逐步生成序列
    for j in range(1, length_session):
        # 获取下一帧的真实数据
        img, last_imgs, actions = dataset[index + j]
        img = img.to(device)
        last_imgs = last_imgs.to(device)
        actions = actions.to(device)
        
        # 添加真实图像到序列
        real_imgs = torch.concat([real_imgs, img[None, :, :, :]], dim=0)
        
        # 使用10步采样生成图像
        if last_imgs.dim() == 4:
            gen_img = model.sample(10, img.shape, gen_10_imgs[-context_length:].unsqueeze(0), actions.unsqueeze(0))[0]
            gen_10_imgs = torch.concat([gen_10_imgs, gen_img[None, :, :, :]], dim=0)
        else:
            prev_frames_input = last_imgs[-context_length:].clone()
            prev_frames_input[:, 0] = gen_10_imgs[-context_length:]
            gen_img = model.sample(10, img.shape, prev_frames_input.unsqueeze(0), actions[-context_length:].unsqueeze(0))[0]
            gen_10_imgs = torch.concat([gen_10_imgs, gen_img[None, :, :, :]], dim=0)
        
        # 使用2步采样生成图像（快速但质量较低）
        if last_imgs.dim() == 4:
            gen_img = model.sample(2, img.shape, gen_2_imgs[-context_length:].unsqueeze(0), actions.unsqueeze(0))[0]
            gen_2_imgs = torch.concat([gen_2_imgs, gen_img[None, :, :, :]], dim=0)
        else:
            prev_frames_input = last_imgs[-context_length:].clone()
            prev_frames_input[:, 0] = gen_2_imgs[-context_length:]
            gen_img = model.sample(2, img.shape, prev_frames_input.unsqueeze(0), actions[-context_length:].unsqueeze(0))[0]
            gen_2_imgs = torch.concat([gen_2_imgs, gen_img[None, :, :, :]], dim=0)
        
        # 使用5步采样生成图像（平衡质量和速度）
        if last_imgs.dim() == 4:
            gen_img = model.sample(5, img.shape, gen_5_imgs[-context_length:].unsqueeze(0), actions.unsqueeze(0))[0]
            gen_5_imgs = torch.concat([gen_5_imgs, gen_img[None, :, :, :]], dim=0)
        else:
            prev_frames_input = last_imgs[-context_length:].clone()
            prev_frames_input[:, 0] = gen_5_imgs[-context_length:]
            gen_img = model.sample(5, img.shape, prev_frames_input.unsqueeze(0), actions[-context_length:].unsqueeze(0))[0]
            gen_5_imgs = torch.concat([gen_5_imgs, gen_img[None, :, :, :]], dim=0)

    # 保存对比图像：真实序列 vs 10步生成 vs 5步生成 vs 2步生成
    base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    out_dir = val_images_dir if os.path.isabs(val_images_dir) else os.path.abspath(os.path.join(base_dir, val_images_dir))
    os.makedirs(out_dir, exist_ok=True)
    _save_sample_imgs(real_imgs, [gen_10_imgs, gen_5_imgs, gen_2_imgs], os.path.join(out_dir, f"{epoch}.png"))

# 命令行接口定义
@click.command()
@click.option('--config', help='训练配置文件路径', metavar='YAML', type=str, required=True, default="config/Diffusion.yaml")
@click.option('--model-type', type=click.Choice(['ddpm', 'edm'], case_sensitive=False), default='edm', help='扩散模型类型：ddpm或edm')
@click.option('--output-prefix', help='输出路径前缀，将与轮次号组合作为完整路径', type=str, required=True)
@click.option('--use-escher-codec', help='使用EscherNet2的编码器条件', is_flag=True, required=False, default=False)

@click.option('--dataset', help='数据集路径', type=str, required=False)
@click.option('--output-loader', help='保存数据加载器的路径', type=str, required=False)
@click.option('--loader', help='预处理好的数据加载器路径', type=str, required=False)
@click.option('--dataset-hourly', help='使用每小时采样数据集（HourlyHailDataset）', is_flag=True, required=False, default=False)
@click.option('--bucket-index', help='每小时选择第几张（0为第一张）', type=int, required=False, default=0)
@click.option('--hourly-loose', help='宽松模式：某小时不足张次时选择最接近的', is_flag=True, required=False, default=False)
@click.option('--dataset-mode', type=click.Choice(['default', 'hourly', 'tiled'], case_sensitive=False), default='default', help='数据集模式：default/hourly/tiled')

@click.option('--gen-val-images', help='是否生成验证图像', is_flag=True, required=False, default=True)

@click.option('--last-checkpoint', help='恢复训练的检查点路径', type=str, required=False)
@click.option('--elevation', help='海拔栅格文件路径（.npy），用于作为动作的额外通道', type=str, required=False)
@click.option('--val-images-dir', help='验证图像保存目录（默认val_images_1）', type=str, required=False, default='val_images_1')
@click.option('--image-size', help='覆盖配置中的图像尺寸', type=int, required=False)
@click.option('--split-by-day', help='按天划分训练/测试（前10天训练，后2天测试）', is_flag=True, required=False, default=False)
@click.option('--use-latent', help='使用SD VAE潜空间训练与采样', is_flag=True, required=False, default=False)
@click.option('--batch-size', help='覆盖训练批次大小', type=int, required=False)
@click.option('--num-workers', help='覆盖数据加载进程数', type=int, required=False)
@click.option('--amp', help='启用混合精度训练(AMP)', is_flag=True, required=False, default=False)
@click.option('--compile', help='启用torch.compile优化', is_flag=True, required=False, default=False)
@click.option('--channels-last', help='启用channels_last内存格式', is_flag=True, required=False, default=False)
@click.option('--tf32', help='启用TF32以提高GPU计算速度', is_flag=True, required=False, default=False)
@click.option('--small-dataset', help='提取小数据集用于调试', is_flag=True, required=False, default=False)
@click.option('--small-dataset-size', help='小数据集大小', type=int, required=False, default=100)
def main(**kwargs):
    """
    主训练函数
    
    这是整个训练流程的入口点，负责：
    1. 解析命令行参数和配置文件
    2. 设置数据预处理和设备
    3. 初始化模型
    4. 准备数据集和数据加载器
    5. 启动训练循环
    """
    # 解析命令行参数
    options = EasyDict(kwargs)
    
    # 加载YAML配置文件
    with open(options.config, 'r') as f:
        config = EasyDict(**yaml.safe_load(f))
    
    # 解析训练和生成配置
    training_config = TrainingConfig(**config.training)
    if options.get('batch_size'):
        training_config.batch_size = int(options.batch_size)
    if options.get('num_workers') is not None:
        training_config.num_workers = int(options.num_workers)
    generation_config = GenerationConfig(**config.generation)
    if options.get('use_latent'):
        generation_config.use_latent = True
    if options.dataset_mode == "tiled":
        try:
            generation_config.neighborhood = 9
        except Exception:
            pass
    else:
        try:
            generation_config.neighborhood = 1
        except Exception:
            pass

    # 定义图像预处理管道
    _img_size = int(options.image_size) if options.get('image_size') else int(generation_config.image_size)
    transform_to_tensor = transforms.Compose([
        transforms.Resize((_img_size, _img_size)),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    ])

    # 自动检测并设置计算设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 对于Mac OS，优先使用MPS（Metal Performance Shaders）
    if torch.backends.mps.is_available():
        device = "mps"
    if device == "cuda":
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            if options.get('tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
    
    # 若启用潜空间但本地VAE不可用，则自动回退到像素空间
    if generation_config.use_latent:
        _vae_probe = VAEAdapter(device=device)
        _vae_probe._try_init()
        if not _vae_probe.available:
            print("警告：未检测到可用的SD VAE，自动回退到像素空间训练与采样")
            generation_config.use_latent = False

    # 根据指定的模型类型初始化扩散模型
    model: BaseDiffusionModel
    if options.model_type == "edm":
        # 初始化EDM（Elucidating the Design Space of Diffusion-Based Generative Models）
        config = EDMConfig(**instantiate_from_config(config.edm))
        model = EDM.from_config(
            config=config,
            context_length=generation_config.context_length,
            device=device,
            model=UNet.from_config(
                config=config.unet,
                in_channels=generation_config.unet_input_channels,  # 输入通道数
                out_channels=(generation_config.latent_channels if generation_config.use_latent else generation_config.output_channels),
                actions_count=generation_config.actions_count,      # 动作数量
                seq_length=generation_config.context_length        # 序列长度
            ),
            use_vision_encoder=options.use_escher_codec,
            use_latent_encoder=generation_config.use_latent
        )
    elif options.model_type == "ddpm":
        # 初始化DDPM（Denoising Diffusion Probabilistic Models）
        config = DDPMConfig(**instantiate_from_config(config.ddpm))
        model = DDPM.from_config(
            config=config,
            context_length=generation_config.context_length,
            device=device,
            model=UNet.from_config(
                config=config.unet,
                in_channels=generation_config.unet_input_channels,
                out_channels=(generation_config.latent_channels if generation_config.use_latent else generation_config.output_channels),
                actions_count=generation_config.actions_count,
                seq_length=generation_config.context_length,
                T=config.T  # DDPM特有的时间步数参数
            )
            ,
            use_latent_encoder=generation_config.use_latent
        )
    if options.get('channels_last'):
        try:
            model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    if options.get('compile'):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # 数据集准备：支持从原始数据或预处理的加载器加载
    if options.dataset:
        # 从原始数据集路径加载
        # 解析海拔路径：优先使用命令行提供，其次默认 data/elevation.npy
        default_elev = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/elevation.npy")
        elev_path = options.get('elevation') if options.get('elevation') else (default_elev if os.path.exists(default_elev) else None)

        # 支持三种数据集模式
        if options.dataset_mode == "tiled":
            wind_root = os.path.join(os.path.dirname(options.dataset), "wind")
            elev_root = os.path.join(os.path.dirname(options.dataset), "elevation")
            dataset = TiledHailDataset(
                images_root=os.path.join(options.dataset),
                wind_root=wind_root,
                elev_root=elev_root if os.path.isdir(elev_root) else None,
                seq_length=generation_config.context_length,
                transform=transform_to_tensor
            )
        elif options.dataset_mode == "hourly" or options.dataset_hourly:
            dataset = HourlyHailDataset(
                images_dir=os.path.join(options.dataset),
                actions_path=os.path.join(options.dataset, "final"),
                seq_length=generation_config.context_length,
                bucket_index=options.bucket_index,
                transform=transform_to_tensor,
                elevation_path=elev_path,
                strict=(not options.hourly_loose),
            )
        else:
            dataset = HailDataset(
                images_dir=os.path.join(options.dataset),
                actions_path=os.path.join(options.dataset, "final"),
                seq_length=generation_config.context_length,
                transform=transform_to_tensor,
                elevation_path=elev_path
            )

        if options.split_by_day:
            days = sorted([d for d in os.listdir(options.dataset) if os.path.isdir(os.path.join(options.dataset, d)) and d.lower() not in {"final", "final_2"}])
            if len(days) >= 2:
                train_days = days[:-2]
                val_days = days[-2:]
            else:
                train_days = days
                val_days = []
            def _seq_day(i: int) -> str:
                imgs, _ = dataset.sequences[i]
                rel = imgs[-1]
                return rel.split('/')[0]
            train_indices = [i for i in range(len(dataset.sequences)) if _seq_day(i) in train_days]
            val_indices = [i for i in range(len(dataset.sequences)) if _seq_day(i) in val_days]
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
        else:
            total_size = len(dataset)
            train_size = int(0.8 * total_size)
            valid_size = total_size - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, valid_size])

        if options.get('small_dataset'):
            n = int(options.small_dataset_size)
            if hasattr(train_dataset, '__len__'):
                tn = min(n, len(train_dataset))
            else:
                tn = n
            if hasattr(val_dataset, '__len__'):
                vn = min(n, len(val_dataset))
            else:
                vn = n
            try:
                train_dataset = Subset(train_dataset, list(range(tn)))
                val_dataset = Subset(val_dataset, list(range(vn)))
                print(f"已启用小数据集调试: 训练集 {tn} 样本, 验证集 {vn} 样本")
            except Exception:
                pass
        # 可选：保存预处理好的数据加载器以便后续使用
        if options.output_loader:
            # 若参数是目录名，则写入该目录下的 loader.pkl；若是以.pkl结尾则按文件路径保存
            output_loader_path = options.output_loader
            if not output_loader_path.endswith('.pkl'):
                os.makedirs(output_loader_path, exist_ok=True)
                output_loader_path = os.path.join(output_loader_path, 'loader.pkl')
            else:
                out_dir = os.path.dirname(output_loader_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
            with open(output_loader_path, 'wb') as f:
                pickle.dump({
                    "train": train_dataset,
                    "val": val_dataset,
                    "all": dataset
                }, f)
    elif options.loader:
        # 从预处理的加载器文件加载
        with open(options.loader, 'rb') as f:
            checkpoint = pickle.load(f)
            train_dataset = checkpoint['train']
            train_dataset.transform = transform_to_tensor
            val_dataset = checkpoint['val']
            val_dataset.transform = transform_to_tensor
            dataset = checkpoint['all']
            dataset.transform = transform_to_tensor

    # 创建训练数据加载器
    dl_args = {
        "batch_size": training_config.batch_size,
        "shuffle": True,
        "num_workers": training_config.num_workers
    }
    if device == "cuda":
        dl_args["pin_memory"] = True
        if training_config.num_workers and training_config.num_workers > 0:
            dl_args["persistent_workers"] = True
            dl_args["prefetch_factor"] = 2
    train_dataloader = DataLoader(
        train_dataset,
        **dl_args
    )
    
    # 创建验证数据加载器
    val_args = {
        "batch_size": training_config.batch_size,
        "num_workers": training_config.num_workers
    }
    if device == "cuda":
        val_args["pin_memory"] = True
        if training_config.num_workers and training_config.num_workers > 0:
            val_args["persistent_workers"] = True
            val_args["prefetch_factor"] = 2
    val_dataloader = DataLoader(
        val_dataset,
        **val_args
    )

    def gen_val_images(epoch: int):
        """验证图像生成回调函数"""
        _generate_and_save_sample_imgs(
            model, dataset, epoch, device, generation_config.context_length,
            val_images_dir=options.val_images_dir
        )

    print(f"开始训练 {options.model_type} 模型")
    
    # 启动训练循环
    train_loop(
        model=model,
        device=device,
        config=training_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        output_path_prefix=options.output_prefix,
        existing_model_path=options["last_checkpoint"],
        gen_imgs=gen_val_images,
        use_amp=bool(options.get('amp')),
        channels_last=bool(options.get('channels_last')),
    )

if __name__ == "__main__":
    main()
