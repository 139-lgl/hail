"""
Q学习智能体模块

该模块实现了一个基于深度Q网络(DQN)的强化学习智能体，用于训练贪吃蛇游戏。
主要功能包括：
- Q网络模型的训练和推理
- 经验回放机制(Experience Replay)
- ε-贪婪策略(ε-greedy policy)
- 训练数据的记录和保存
- 模型检查点的保存和加载

核心组件：
1. QAgentConfig: 智能体配置参数
2. ReplayMemory: 经验回放缓冲区
3. QAgent: 主要的Q学习智能体类
4. ValueForEndGame: 游戏结束时的处理策略

特性：
- 支持自动保存训练快照和动作序列
- 可配置的训练参数和策略
- 支持从检查点恢复训练
- 实时训练进度可视化
"""

from collections import deque
from typing import Union, Tuple, List, Optional
import random
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shutil

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from models.agent.blocks import Linear_QNet, QTrainer
from game.env import Environment, ActionResult

def plot(scores, mean_scores):
    """
    绘制训练过程中的得分曲线
    
    Args:
        scores: 每局游戏的得分列表
        mean_scores: 平均得分列表
    
    功能：
    - 清除之前的图像
    - 绘制得分和平均得分曲线
    - 显示最新的得分值
    - 实时更新显示
    """
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

class ValueForEndGame(Enum):
    """
    游戏结束时的处理策略枚举
    
    - last_action: 在游戏结束时记录最后一个动作
    - not_exist: 游戏结束时不记录额外动作
    """
    last_action = "last_action"
    not_exist = "not_exist"

@dataclass
class QAgentConfig:
    """
    Q学习智能体的配置参数
    
    Args:
        max_memory: 经验回放缓冲区的最大容量
        batch_size: 训练时的批次大小
        lr: 学习率
        hidden_state: 神经网络隐藏层大小
        value_for_end_game: 游戏结束时的处理策略
        iterations: 总训练迭代次数
        min_deaths_to_record: 开始记录数据前的最小死亡次数
        epsilon_start: ε-贪婪策略的初始探索率
        epsilon_min: ε-贪婪策略的最小探索率
        epsilon_decay: ε-贪婪策略的衰减率
        gamma: 折扣因子，用于计算未来奖励的权重
        train_every_iteration: 每隔多少次迭代进行一次训练
        save_every_iteration: 每隔多少次迭代保存一次模型(可选)
    """
    max_memory: int
    batch_size: int
    lr: float
    hidden_state: int
    value_for_end_game: ValueForEndGame
    iterations: int
    min_deaths_to_record: int
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    gamma: float = 0.9
    train_every_iteration: int = 10
    save_every_iteration: Optional[int] = None

class ReplayMemory:
    """
    经验回放缓冲区
    
    用于存储和采样训练经验，实现经验回放机制以提高训练稳定性。
    使用双端队列(deque)实现固定大小的循环缓冲区。
    """
    
    def __init__(self, capacity: int):
        """
        初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区最大容量
        """
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        向缓冲区添加一条经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """
        从缓冲区随机采样一批经验
        
        Args:
            batch_size: 采样的批次大小
            
        Returns:
            采样的经验批次，按类型分组返回
        """
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        return zip(*batch)
    
    def __len__(self):
        """返回缓冲区中经验的数量"""
        return len(self.memory)

class QAgent:
    """
    Q学习智能体
    
    实现基于深度Q网络的强化学习智能体，用于训练贪吃蛇游戏。
    包含经验回放、ε-贪婪策略、模型训练和数据记录等功能。
    """
    
    def __init__(
        self,
        env: Environment,
        config: QAgentConfig,
        model_path: str,
        dataset_path: str,
        last_checkpoint: Optional[str]
    ):
        """
        初始化Q学习智能体
        
        Args:
            env: 游戏环境
            config: 智能体配置
            model_path: 模型保存路径
            dataset_path: 数据集保存路径
            last_checkpoint: 上次检查点路径(可选，用于恢复训练)
        """
        self.config = config
        self.model_path = model_path
        self.memory = ReplayMemory(config.max_memory)
        # 创建Q网络：输入为状态维度，输出为动作数量
        self.model = Linear_QNet(len(env.get_state()), self.config.hidden_state, env.actions_length())
        self.trainer = QTrainer(self.model, gamma=config.gamma)
        self.env = env
        self.steps = 0
        self.dataset_path = dataset_path
        self.count_games = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.recorded_actions = []
        self.epsilon = config.epsilon_start
        self.begin_iteration = 0
        
        # 如果提供了检查点，则加载之前的训练状态
        if last_checkpoint:
            parameters = torch.load(last_checkpoint)
            self.model.load_state_dict(parameters["model"])
            self.optimizer.load_state_dict(parameters["optimizer"])
            self.count_games = parameters.get("count_games", 0)
            self.begin_iteration = parameters.get("begin_iteration", 0)
    
    def _remember(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, List[int]],
        reward: Union[np.ndarray, float],
        next_state: np.ndarray,
        done: Union[np.ndarray, bool]
    ):
        """
        将经验存储到回放缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.memory.append((state, action, reward, next_state, done))

    def _train_long_memory(self):
        """
        使用经验回放进行长期记忆训练
        
        从回FFER区采样一批经验进行训练，如果缓冲区大小不足批次大小，
        则使用所有可用的经验。
        """
        if len(self.memory) > self.config.batch_size:
            mini_sample = random.sample(self.memory, self.config.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self._train_step(states, actions, rewards, next_states, dones)

    def _train_step(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, List[int]],
        reward: Union[np.ndarray, float],
        next_state: np.ndarray,
        done: Union[np.ndarray, bool]
    ):
        """
        执行一步训练
        
        Args:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
            
        功能：
        - 清零梯度
        - 计算损失
        - 反向传播
        - 更新参数
        """
        self.optimizer.zero_grad()
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        loss.backward()
        self.optimizer.step()

    @property
    def snapshots_path(self):
        """获取快照保存路径"""
        return os.path.join(self.dataset_path, "snapshots")

    @property
    def actions_path(self):
        """获取动作序列保存路径"""
        return os.path.join(self.dataset_path, "actions")

    def _get_action(self, state: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        根据当前状态选择动作
        
        Args:
            state: 当前游戏状态
            
        Returns:
            (action_vector, action_index): 动作向量和动作索引
            
        策略：
        - 使用ε-贪婪策略
        - 以ε概率随机选择动作(探索)
        - 以(1-ε)概率选择Q值最大的动作(利用)
        """
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            max_index = random.randint(0, self.env.actions_length() - 1)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float)
                q_values = self.model(state_tensor)
                max_index = torch.argmax(q_values).item()
        
        # 将动作索引转换为one-hot编码
        final_move = [0] * self.env.actions_length()
        final_move[max_index] = 1
        return np.array(final_move), max_index
    
    def _save_snapshot(self, step: int):
        """
        保存当前游戏画面快照
        
        Args:
            step: 当前步数，用作文件名
        """
        plt.imsave(os.path.join(self.snapshots_path, f'{step}.jpg'), self.env.get_snapshot())
    
    def _save_actions(self):
        """
        保存记录的动作序列到文件
        
        将所有记录的动作索引保存为文本文件，每行一个动作。
        """
        with open(self.actions_path, mode="w") as file:
            file.write("\n".join([str(action) for action in self.recorded_actions]))
    
    def play_step(
        self,
        record: bool = False,
        step: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, ActionResult]:
        """
        执行一步游戏
        
        Args:
            record: 是否记录游戏数据(快照和动作)
            step: 指定的步数(可选)
            
        Returns:
            (old_state, action, result): 旧状态、执行的动作和游戏结果
            
        功能：
        - 获取当前状态
        - 选择并执行动作
        - 可选地记录游戏数据
        - 返回状态转换信息
        """
        old_state = self.env.get_state()
        action, max_index = self._get_action(old_state)
        self.steps += 1
        if step is None:
            step = self.steps
        result = self.env.do_action(action)
        
        # 如果需要记录，保存快照和动作
        if record:
            self._save_snapshot(step)
            self.recorded_actions.append(max_index)
            self._save_actions()
        return old_state, action, result

    def train(self, show_plot: bool = False, record: bool = False, clear_old: bool = False):
        """
        训练智能体
        
        Args:
            show_plot: 是否显示训练进度图表
            record: 是否记录训练数据
            clear_old: 是否清除旧的训练数据
            
        功能：
        - 设置训练环境
        - 执行指定次数的训练迭代
        - 使用ε-贪婪策略进行探索和利用
        - 定期进行模型训练
        - 保存最佳模型和检查点
        - 可选地显示训练进度和记录数据
        """
        self._setup_training(clear_old)
        
        plot_scores = []
        plot_mean_scores = []
        top_result = 0
        total_score = 0
        print(f"Begin iteration is {self.begin_iteration}")
        print(f"All iteration is {self.config.iterations}")
        if self.begin_iteration >= self.config.iterations:
            return
            
        for iteration in range(self.begin_iteration, self.config.iterations):
            # 执行一步游戏，如果游戏次数足够则开始记录
            old_state, action, result = self.play_step(
                record=record and self.count_games >= self.config.min_deaths_to_record
            )
            reward, new_state, done = result.reward, result.new_state, result.terminated
            
            # 将经验存储到回放缓冲区
            self.memory.push(old_state, action, result.reward, result.new_state, result.terminated)

            def do_training():
                """执行一次训练"""
                batch = self.memory.sample(self.config.batch_size)
                self._train_step(*batch)

            # 定期进行训练
            if len(self.memory) > self.config.batch_size and iteration % self.config.train_every_iteration == 0:
                do_training()
            
            # 衰减探索率
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

            # 游戏结束时的处理
            if done:
                self.count_games += 1
                score = result.score
                self.env.reset()
                do_training()
                
                # 根据配置处理游戏结束时的记录
                if record and self.count_games > self.config.min_deaths_to_record:
                    if self.config.value_for_end_game.value == ValueForEndGame.last_action.value:
                        # 记录最后一个动作(游戏结束动作)
                        self.steps += 1
                        self.recorded_actions.append(self.env.actions_length())
                        self._save_snapshot(self.steps)
                    elif self.config.value_for_end_game.value == ValueForEndGame.not_exist.value:
                        # 不记录额外动作
                        pass
                self._save_actions()

                # 保存最佳模型
                if score > top_result:
                    top_result = score
                    self.save_agent(iteration)

                print('Game', self.count_games, 'Score', score, 'Record:', top_result, "Iteration:", iteration)
                
                # 更新训练进度图表
                if show_plot:
                    plot_scores.append(score)
                    total_score += score
                    mean_score = total_score / self.count_games
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)
            
            # 定期保存检查点
            if self.config.save_every_iteration is not None and iteration % self.config.save_every_iteration == 0:
                self.save_agent(iteration)
        
        # 训练结束后的清理工作
        self._save_actions()
        self.save_agent(iteration+1)
        print(f"finish iteration is {iteration}")

    def _setup_training(self, clear_old: bool):
        """
        设置训练环境
        
        Args:
            clear_old: 是否清除旧数据
            
        功能：
        - 根据参数决定是清除还是加载旧数据
        - 创建必要的目录
        """
        if clear_old:
            self._clear_training_data()
        else:
            self._load_training_data()
        os.makedirs(self.snapshots_path, exist_ok=True)
        if os.path.dirname(self.model_path) != "":
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def _clear_training_data(self):
        """
        清除训练数据
        
        重置步数和动作记录，删除数据集目录。
        """
        self.steps = 0
        self.recorded_actions = []
        shutil.rmtree(self.dataset_path)

    def _load_training_data(self):
        """
        加载已有的训练数据
        
        功能：
        - 从快照目录计算已有步数
        - 从动作文件加载动作序列
        - 验证数据一致性
        - 如果加载失败则重置为初始状态
        """
        try:
            # 计算已保存的快照数量
            self.steps = len([f for f in os.listdir(self.snapshots_path) if f.endswith('.jpg')])
            # 加载动作序列
            with open(self.actions_path) as f:
                self.recorded_actions = [int(line) for line in f]
        except:
            # 如果加载失败，重置为初始状态
            self.steps = 0
            self.recorded_actions = []
        print(self.steps, len(self.recorded_actions))
        # 确保步数和动作数量一致
        assert self.steps == len(self.recorded_actions)

    def save_agent(self, iteration: int):
        """
        保存智能体状态
        
        Args:
            iteration: 当前迭代次数
            
        保存内容：
        - 模型参数
        - 优化器状态
        - 游戏计数
        - 开始迭代次数
        """
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "count_games": self.count_games,
            "begin_iteration": iteration
        }, self.model_path)