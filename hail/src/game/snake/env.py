"""
贪吃蛇游戏环境包装器

这个模块为贪吃蛇游戏提供了标准化的强化学习环境接口，具有以下特点：
1. 实现了Environment基类的标准接口
2. 提供状态表示和动作处理
3. 计算奖励和终止条件
4. 支持Q学习智能体训练

主要功能：
- 状态编码：将游戏状态转换为11维特征向量
- 动作处理：处理4个方向的移动动作
- 奖励设计：吃到食物+10，游戏结束-10，超时-10
- 终止条件：碰撞死亡或超过最大步数

作者：Snake Diffusion项目组
"""

from typing import Tuple, Union, List

import numpy as np

from ..env import Environment, ActionResult
from .game import SnakeGame, Direction, Point

class GameEnvironment(Environment):
    """
    贪吃蛇游戏环境类
    
    将SnakeGame包装成标准的强化学习环境，提供：
    - 状态观察：11维特征向量（危险检测+方向+食物位置）
    - 动作空间：4个离散动作（上下左右）
    - 奖励函数：基于游戏表现的奖励设计
    - 终止条件：死亡或超时
    """
    
    def __init__(self, game: SnakeGame):
        """
        初始化游戏环境
        
        Args:
            game: SnakeGame实例
        """
        super().__init__()
        self.steps_taken = 0    # 记录已执行的步数
        self.game = game        # 游戏实例
        game.reset()            # 重置游戏状态

    def actions_length(self) -> int:
        """
        返回动作空间大小
        
        Returns:
            动作数量（4个方向）
        """
        return 4

    def reset(self):
        """
        重置环境到初始状态
        
        重置游戏和步数计数器
        """
        self.game.reset()
        self.steps_taken = 0

    def get_snapshot(self) -> np.ndarray:
        """
        获取游戏画面快照
        
        用于视觉观察或数据收集
        
        Returns:
            转置后的游戏画面数组 [height, width, channels]
        """
        return self.game.get_snapshot().transpose(1,0,2)

    def do_action(self, action: np.ndarray) -> ActionResult:
        """
        执行动作并返回结果
        
        这是环境的主要接口，处理动作执行和状态转换
        
        Args:
            action: 动作数组（通常是one-hot编码或动作索引）
            
        Returns:
            ActionResult: 包含新状态、奖励、终止标志和分数的结果对象
        """
        self.steps_taken += 1
        
        # 在游戏中执行动作
        reward, terminated = self._take_action(action)
        return ActionResult(self.get_state(), reward, terminated, self.game.score)
    
    def get_state(self) -> np.ndarray:
        """
        获取当前游戏状态的特征表示
        
        将游戏状态编码为11维特征向量：
        - 前3维：危险检测（直行、右转、左转方向的危险）
        - 中4维：当前移动方向（one-hot编码）
        - 后4维：食物相对位置（左、右、上、下）
        
        Returns:
            状态特征向量 [11,]
        """
        head = self.game.head
        
        # 计算蛇头周围四个方向的位置
        point_l = Point(head.x - self.game.block_size, head.y)  # 左侧位置
        point_r = Point(head.x + self.game.block_size, head.y)  # 右侧位置
        point_u = Point(head.x, head.y - self.game.block_size)  # 上方位置
        point_d = Point(head.x, head.y + self.game.block_size)  # 下方位置
        
        # 获取当前移动方向
        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        state = [
            # 危险检测1：直行方向是否有危险
            (dir_r and self.game.is_collision(point_r)) or 
            (dir_l and self.game.is_collision(point_l)) or 
            (dir_u and self.game.is_collision(point_u)) or 
            (dir_d and self.game.is_collision(point_d)),

            # 危险检测2：右转方向是否有危险
            (dir_u and self.game.is_collision(point_r)) or 
            (dir_d and self.game.is_collision(point_l)) or 
            (dir_l and self.game.is_collision(point_u)) or 
            (dir_r and self.game.is_collision(point_d)),

            # 危险检测3：左转方向是否有危险
            (dir_d and self.game.is_collision(point_r)) or 
            (dir_u and self.game.is_collision(point_l)) or 
            (dir_r and self.game.is_collision(point_u)) or 
            (dir_l and self.game.is_collision(point_d)),
            
            # 当前移动方向（one-hot编码）
            dir_l,    # 是否向左
            dir_r,    # 是否向右
            dir_u,    # 是否向上
            dir_d,    # 是否向下
            
            # 食物相对位置
            self.game.food.x < head.x,  # 食物在左侧
            self.game.food.x > head.x,  # 食物在右侧
            self.game.food.y < head.y,  # 食物在上方
            self.game.food.y > head.y   # 食物在下方
        ]

        return np.array(state, dtype=int)

    def _take_action(self, action: np.ndarray) -> Tuple[int, bool]:
        """
        执行动作并计算奖励
        
        处理动作执行、奖励计算和终止条件判断
        
        Args:
            action: 要执行的动作
            
        Returns:
            tuple: (奖励值, 是否终止)
        """
        prev_score = self.game.score
        game_over, score = self.game.play_step(action)
        
        # 奖励计算
        reward = 0
        if game_over:
            reward = -10        # 游戏结束惩罚
        elif score > prev_score:
            reward = 10         # 吃到食物奖励
            
        # 超时检测：如果步数超过蛇长度的100倍，强制结束
        if self.steps_taken >= 100 * len(self.game.snake):
            game_over = True
            self.game.draw_game_over()
            reward = -10        # 超时惩罚
            
        return reward, game_over

if __name__ == '__main__':
    """
    测试代码
    
    创建游戏环境并测试基本功能
    """
    game = SnakeGame()
    env = GameEnvironment(game)
    state = env.get_state()  # 修正方法名
    print(state.shape)
    print(env.observation_space)
    print(env.action_space)