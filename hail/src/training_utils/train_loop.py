"""
扩散模型训练循环模块

这个模块实现了扩散模型的核心训练循环，包括：
1. 训练配置管理
2. 模型训练和验证
3. 损失计算和优化
4. 检查点保存和恢复
5. 验证图像生成

作者：Snake Diffusion项目组
"""

from typing import Optional, Callable
import random
from dataclasses import dataclass
import os

import torch
import tqdm
from torch.utils.data.dataloader import DataLoader

from models.gen.blocks import BaseDiffusionModel

@dataclass
class TrainingConfig:
    """
    训练配置类
    
    包含训练过程中需要的所有超参数和配置选项
    
    Attributes:
        epochs: 训练轮次数
        batch_size: 批次大小
        num_workers: 数据加载器的工作进程数
        save_every_epoch: 每隔多少轮次保存一次模型（None表示不定期保存）
    """
    epochs: int
    batch_size: int
    num_workers: int
    save_every_epoch: Optional[int] = None

def train_loop(
    model: BaseDiffusionModel,
    device: str,
    config: TrainingConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    output_path_prefix: str,
    existing_model_path: Optional[str] = None,
    gen_imgs: Callable[[int], None] = None,
    action_change: Optional[bool] = False,
    use_amp: bool = False,
    channels_last: bool = False
):
    """
    扩散模型训练主循环
    
    这是整个训练过程的核心函数，负责：
    1. 初始化优化器和训练状态
    2. 从检查点恢复训练（如果提供）
    3. 执行训练和验证循环
    4. 保存模型检查点
    5. 生成验证图像（如果启用）
    
    Args:
        model: 要训练的扩散模型
        device: 计算设备（cuda/cpu/mps）
        config: 训练配置
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        output_path_prefix: 输出文件路径前缀
        existing_model_path: 现有模型检查点路径（用于恢复训练）
        gen_imgs: 验证图像生成回调函数
        action_change: 是否启用动作变化（暂未使用）
    
    Returns:
        tuple: (训练损失列表, 验证损失列表)
    """
    # 创建输出目录
    if os.path.dirname(output_path_prefix) != "":
        os.makedirs(os.path.dirname(output_path_prefix), exist_ok=True)
    
    # 创建验证图像目录（如果需要生成验证图像）
    if gen_imgs:
        os.makedirs("val_images", exist_ok=True)
    
    # 初始化Adam优化器，学习率设为2e-4（扩散模型的常用学习率）
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))
    
    # 设置训练轮次范围
    epoch_range = range(1, config.epochs + 1)
    
    # 如果提供了现有模型路径，则从检查点恢复训练
    if existing_model_path is not None:
        print(f"从检查点恢复训练: {existing_model_path}")
        parameters = torch.load(existing_model_path, map_location=device)
        model.load_state_dict(parameters["model"])        # 恢复模型参数
        optimizer.load_state_dict(parameters["optimizer"]) # 恢复优化器状态
        # 从上次停止的轮次继续训练
        epoch_range = range(parameters["epoch"] + 1, config.epochs + 1)

    # 初始化损失记录列表
    training_losses = []
    val_losses = []
    
    # 开始训练循环
    iteration_count = 0
    max_log_iterations = 100
    for epoch in epoch_range:
        # 设置模型为训练模式
        model.train(True)
        training_loss = 0
        val_loss = 0
        
        # 训练阶段
        print(f"开始第 {epoch} 轮训练...")
        pbar = tqdm.tqdm(train_dataloader, desc=f"训练轮次 {epoch}")
        
        for index, (imgs, previous_frames, previous_actions) in enumerate(pbar):
            # 清零梯度
            optimizer.zero_grad()
            
            # 将数据移动到指定设备
            imgs = imgs.to(device, non_blocking=True)
            previous_frames = previous_frames.to(device, non_blocking=True)
            previous_actions = previous_actions.to(device, non_blocking=True)
            if channels_last:
                imgs = imgs.contiguous(memory_format=torch.channels_last)
    
            iteration_count += 1
            if iteration_count <= max_log_iterations:
                try:
                    print(f"Iteration {iteration_count}:")
                    print(f"imgs: {tuple(imgs.shape)}")
                    print(f"previous_frames: {tuple(previous_frames.shape)}")
                    print(f"previous_actions: {tuple(previous_actions.shape)}")
                    if torch.cuda.is_available():
                        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                        print(f"cuda_max_memory_allocated: {mem_mb:.2f} MB")
                except Exception:
                    pass

            # 前向传播：计算扩散模型的损失
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    loss = model.forward(imgs, previous_frames, previous_actions)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.forward(imgs, previous_frames, previous_actions)
                loss.backward()
                optimizer.step()
    
            # 累积训练损失
            training_loss += loss.item()
            
            # 更新进度条显示
            pbar.set_description(f"训练轮次 {epoch} - 损失: {training_loss / (index + 1):.4f}")
        
        # 验证阶段
        print(f"开始第 {epoch} 轮验证...")
        model.eval()  # 设置模型为评估模式
        
        with torch.no_grad():  # 验证时不计算梯度
            val_pbar = tqdm.tqdm(val_dataloader, desc=f"验证轮次 {epoch}")
            
            for index, (imgs, previous_frames, previous_actions) in enumerate(val_pbar):
                # 将数据移动到指定设备
                imgs = imgs.to(device, non_blocking=True)
                previous_frames = previous_frames.to(device, non_blocking=True)
                previous_actions = previous_actions.to(device, non_blocking=True)
                if channels_last:
                    imgs = imgs.contiguous(memory_format=torch.channels_last)
    
                # 计算验证损失（不进行反向传播）
                if use_amp and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        loss = model(imgs, previous_frames, previous_actions)
                else:
                    loss = model(imgs, previous_frames, previous_actions)
        
                # 累积验证损失
                val_loss += loss.item()
                
                # 更新进度条显示
                val_pbar.set_description(f"验证轮次 {epoch} - 验证损失: {val_loss / (index + 1):.4f}")
            
            # 生成验证图像（如果启用）
            if gen_imgs:
                print(f"生成第 {epoch} 轮验证图像...")
                gen_imgs(epoch)
        
        # 记录平均损失
        avg_train_loss = training_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        
        training_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"轮次 {epoch} 完成 - 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        
        # 定期保存模型检查点
        if config.save_every_epoch is not None and epoch > 0 and epoch % config.save_every_epoch == 0:
            checkpoint_path = f"{output_path_prefix}_{epoch}.pth"
            print(f"保存检查点: {checkpoint_path}")
            torch.save({
                "model": model.state_dict(),      # 模型参数
                "optimizer": optimizer.state_dict(), # 优化器状态
                "epoch": epoch                    # 当前轮次
            }, checkpoint_path)
    
    print("训练完成！")
    return training_losses, val_losses
