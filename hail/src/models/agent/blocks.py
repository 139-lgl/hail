
"""
Q学习智能体神经网络模块

该模块定义了Q学习智能体使用的神经网络架构和训练器。
主要组件包括：
- Linear_QNet: 线性Q网络，用于估计状态-动作价值函数
- QTrainer: Q学习训练器，实现Q学习算法的核心训练逻辑

核心功能：
1. Q值估计：根据游戏状态预测每个动作的Q值
2. Q学习训练：使用Bellman方程更新Q网络参数
3. 目标Q值计算：基于奖励和未来状态的最大Q值

特性：
- 简单的全连接网络架构
- 支持批量训练
- 使用MSE损失函数
- 实现标准的Q学习更新规则
"""

import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Linear_QNet(nn.Module):
    """
    线性Q网络
    
    一个简单的全连接神经网络，用于估计状态-动作价值函数(Q函数)。
    网络结构：输入层 -> 隐藏层(ReLU激活) -> 输出层
    
    Args:
        input_size: 输入状态的维度
        hidden_size: 隐藏层的神经元数量
        output_size: 输出动作的数量
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        初始化线性Q网络
        
        Args:
            input_size: 输入状态向量的维度
            hidden_size: 隐藏层的大小
            output_size: 输出动作的数量(Q值的数量)
        """
        super().__init__()
        # 第一层：输入层到隐藏层
        self.linear1 = nn.Linear(input_size, hidden_size)
        # 第二层：隐藏层到输出层
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入状态张量，形状为 (batch_size, input_size) 或 (input_size,)
            
        Returns:
            Q值张量，形状为 (batch_size, output_size) 或 (output_size,)
            每个元素表示对应动作的Q值
        """
        # 通过第一层并应用ReLU激活函数
        x = F.relu(self.linear1(x))
        # 通过第二层输出Q值(无激活函数)
        x = self.linear2(x)
        return x


class QTrainer:
    """
    Q学习训练器
    
    实现Q学习算法的训练逻辑，使用Bellman方程更新Q网络参数。
    核心思想：Q(s,a) = r + γ * max(Q(s',a'))
    
    Args:
        model: 要训练的Q网络模型
        gamma: 折扣因子，控制未来奖励的重要性
    """
    
    def __init__(self, model: torch.nn.Module, gamma: float):
        """
        初始化Q学习训练器
        
        Args:
            model: Q网络模型
            gamma: 折扣因子，范围[0,1]，越接近1越重视长期奖励
        """
        self.gamma = gamma
        self.model = model
        # 使用均方误差损失函数
        self.criterion = nn.MSELoss()

    def train_step(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, int],
        reward: Union[np.ndarray, float],
        next_state: np.ndarray,
        done: Union[np.ndarray, bool]
    ) -> torch.Tensor:
        """
        执行一步Q学习训练
        
        Args:
            state: 当前状态，可以是单个状态或批量状态
            action: 执行的动作，one-hot编码或动作索引
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否为终止状态
            
        Returns:
            计算得到的损失值
            
        算法流程：
        1. 将输入转换为PyTorch张量
        2. 使用当前网络预测当前状态的Q值
        3. 计算目标Q值：Q_target = reward + γ * max(Q(next_state))
        4. 更新对应动作的Q值
        5. 计算并返回MSE损失
        """
        # 将numpy数组转换为PyTorch张量
        state = torch.tensor(np.array(state, dtype=np.float32), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state, dtype=np.float32), dtype=torch.float)
        action = torch.tensor(np.array(action, dtype=np.int64), dtype=torch.long)
        reward = torch.tensor(np.array(reward, dtype=np.float32), dtype=torch.float)
        
        # 处理单个样本的情况，添加批次维度
        if len(state.shape) == 1:
            # (1, x) - 添加批次维度
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. 使用当前网络预测当前状态的Q值
        pred = self.model(state)

        # 2. 创建目标Q值张量(初始化为预测值)
        target = pred.clone()
        
        # 3. 为每个样本计算目标Q值
        for idx in range(len(done)):
            # 基础奖励
            Q_new = reward[idx]
            
            # 如果不是终止状态，添加折扣的未来最大Q值
            if not done[idx]:
                # Q_new = r + γ * max(Q(s', a'))
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # 更新对应动作的目标Q值
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        # 4. 计算预测Q值和目标Q值之间的MSE损失
        return self.criterion.forward(target, pred)