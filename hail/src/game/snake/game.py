"""
贪吃蛇游戏实现

这个模块实现了经典的贪吃蛇游戏，具有以下特点：
1. 基于Pygame的图形界面
2. 支持键盘和AI智能体控制
3. 可配置的游戏参数（速度、大小等）
4. 提供游戏状态快照功能
5. 碰撞检测和游戏结束逻辑

这个游戏环境主要用于：
- Q学习智能体的训练环境
- 生成训练数据用于扩散模型
- 人工游戏体验

作者：Snake Diffusion项目组
"""

import random
from typing import Optional, Union, List
from enum import Enum
from dataclasses import dataclass

import pygame
import numpy as np

# 初始化Pygame
pygame.init()

# 定义游戏常量
SPEED = 10                    # 默认游戏速度
WHITE = (255, 255, 255)      # 白色
RED = (200, 0, 0)            # 红色（食物）
BLUE1 = (0, 0, 255)          # 蓝色1（蛇身外层）
BLUE2 = (0, 100, 255)        # 蓝色2（蛇身内层）
BLACK = (0, 0, 0)            # 黑色（背景）

class Direction(Enum):
    """
    方向枚举类
    
    定义蛇可以移动的四个方向
    """
    RIGHT = 0    # 向右
    LEFT = 1     # 向左
    UP = 2       # 向上
    DOWN = 3     # 向下

@dataclass
class Point:
    """
    点坐标类
    
    表示游戏中的二维坐标点，用于蛇身和食物的位置
    
    Attributes:
        x: x坐标
        y: y坐标
    """
    x: int
    y: int

    def equal_with_block(self, other: 'Point', block_size: int) -> bool:
        """
        检查两个点是否在同一个方块内
        
        考虑到方块大小，判断两个点是否重叠
        
        Args:
            other: 另一个点
            block_size: 方块大小
            
        Returns:
            是否重叠
        """
        return (other.x >= self.x and other.x <= self.x + block_size / 2 and other.y >= self.y and other.y <= self.y + block_size / 2) or \
        (self.x >= other.x and self.x <= other.x + block_size / 2 and self.y >= other.y and self.y <= other.y + block_size / 2)

    def distance(self, other: 'Point') -> float:
        """
        计算两点之间的欧几里得距离
        
        Args:
            other: 另一个点
            
        Returns:
            两点间距离
        """
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

class SnakeGame:
    """
    贪吃蛇游戏主类
    
    实现完整的贪吃蛇游戏逻辑，包括：
    - 游戏初始化和重置
    - 蛇的移动和生长
    - 食物生成和消费
    - 碰撞检测
    - 游戏状态管理
    - 图形界面渲染
    """
    
    def __init__(self, width=640, height=480, speed=SPEED, block_size=20):
        """
        初始化游戏
        
        Args:
            width: 游戏窗口宽度
            height: 游戏窗口高度
            speed: 游戏速度（FPS）
            block_size: 方块大小（像素）
        """
        self.width = width
        self.height = height
        
        # 初始化显示窗口
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.speed = speed

        self.block_size = block_size
        
        # 初始化游戏状态
        self.reset()

        # 用于调试的代码（已注释）
        # import os
        # import matplotlib.pyplot as plt
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # plt.imsave(os.path.join(dir_path, f'test.jpg'), self.get_snapshot().transpose(1,0,2))
    
    def reset(self):
        """
        重置游戏到初始状态
        
        重新初始化蛇的位置、方向、分数和食物
        """
        # 初始化蛇的方向（向右）
        self.direction = Direction.RIGHT
        
        # 初始化蛇头位置（屏幕中央）
        self.head = Point(self.width/2, self.height/2)
        
        # 初始化蛇身（3个方块，向左延伸）
        self.snake = [
            self.head,
            Point(self.head.x-self.block_size, self.head.y),
            Point(self.head.x-(2*self.block_size), self.head.y)
        ]
        
        # 重置分数
        self.score = 0
        self.food = None
        self._place_food()    # 放置第一个食物
        self._update_ui()     # 更新界面

    def get_snapshot(self) -> np.ndarray:
        """
        获取当前游戏画面的像素数组
        
        用于AI训练时获取游戏状态的视觉表示
        
        Returns:
            游戏画面的RGB像素数组 [width, height, 3]
        """
        return pygame.surfarray.array3d(self.display)
        
    def _place_food(self):
        """
        在随机位置放置食物
        
        确保食物不会出现在蛇身上
        """
        # 在网格上随机选择位置
        x = random.randint(0, (self.width-self.block_size)//self.block_size)*self.block_size
        y = random.randint(0, (self.height-self.block_size)//self.block_size)*self.block_size
        self.food = Point(x, y)
        
        # 如果食物出现在蛇身上，重新放置
        if self.food in self.snake:
            self._place_food()

    def _get_direction_from_event(self, event: pygame.event.Event) -> Direction:
        """
        从键盘事件获取移动方向
        
        处理用户键盘输入，防止蛇反向移动
        
        Args:
            event: Pygame事件
            
        Returns:
            新的移动方向
        """
        if event.type != pygame.KEYDOWN:
            return self.direction
            
        # 检查按键并防止反向移动
        if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
            return Direction.LEFT
        elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
            return Direction.RIGHT
        elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
            return Direction.UP
        elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
            return Direction.DOWN
        return self.direction
    
    def _get_direction_from_int_value(self, value: int) -> Direction:
        """
        从整数值获取移动方向
        
        用于AI智能体控制，将动作编号转换为方向
        
        Args:
            value: 动作编号（0-3对应四个方向）
            
        Returns:
            新的移动方向
        """
        new_direction = Direction(value)
        
        # 防止反向移动
        if new_direction == Direction.LEFT and self.direction != Direction.RIGHT:
            return Direction.LEFT
        elif new_direction == Direction.RIGHT and self.direction != Direction.LEFT:
            return Direction.RIGHT
        elif new_direction == Direction.UP and self.direction != Direction.DOWN:
            return Direction.UP
        elif new_direction == Direction.DOWN and self.direction != Direction.UP:
            return Direction.DOWN
        return self.direction
        
        # 以下是另一种实现方式（已注释）：基于相对转向
        # clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # idx = clock_wise.index(self.direction)
        # if value == 0:
        #     return self.direction
        # elif value == 1:
        #     next_idx = (idx + 1) % 4
        #     return clock_wise[next_idx]
        # else:
        #     next_idx = (idx - 1) % 4
        #     return clock_wise[next_idx]
    
    def _get_direction_from_list_int_value(self, value: np.ndarray) -> Direction:
        """
        从数组值获取移动方向
        
        用于处理神经网络输出的概率分布
        
        Args:
            value: 动作概率数组
            
        Returns:
            新的移动方向
        """
        # 选择概率最大的动作
        value = value.argmax()
        if value > 3:
            return self.direction
            
        new_direction = Direction(value)
        
        # 防止反向移动
        if new_direction == Direction.LEFT and self.direction != Direction.RIGHT:
            return Direction.LEFT
        elif new_direction == Direction.RIGHT and self.direction != Direction.LEFT:
            return Direction.RIGHT
        elif new_direction == Direction.UP and self.direction != Direction.DOWN:
            return Direction.UP
        elif new_direction == Direction.DOWN and self.direction != Direction.UP:
            return Direction.DOWN
        return self.direction
        
        # 以下是另一种实现方式（已注释）：基于one-hot编码
        # clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # idx = clock_wise.index(self.direction)
        # if np.array_equal(value, [1, 0, 0]):
        #     new_dir = clock_wise[idx] # no change
        # elif np.array_equal(value, [0, 1, 0]):
        #     next_idx = (idx + 1) % 4
        #     new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        # else: # [0, 0, 1]
        #     next_idx = (idx - 1) % 4
        #     new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
        # return new_dir
    
    def play_step(self, value: Optional[Union[int, List[int]]] = None):
        """
        执行一个游戏步骤
        
        这是游戏的主要逻辑函数，处理：
        1. 用户输入或AI动作
        2. 蛇的移动
        3. 食物消费和蛇的生长
        4. 碰撞检测
        5. 界面更新
        
        Args:
            value: 可选的动作输入（用于AI控制）
                  - None: 使用键盘输入
                  - int: 动作编号
                  - List[int]: 动作概率分布
                  
        Returns:
            tuple: (game_over, score) 游戏是否结束和当前分数
        """
        # 1. 收集用户输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            self.direction = self._get_direction_from_event(event)
            
        # 处理AI输入
        if value is not None:
            if isinstance(value, int) or isinstance(value, np.int64):
                self.direction = self._get_direction_from_int_value(value)
            else:
                self.direction = self._get_direction_from_list_int_value(value)
                
        # 2. 移动蛇头
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # 3. 检查是否吃到食物
        if self.head.equal_with_block(self.food, self.block_size):
            self.score += 1      # 增加分数
            self._place_food()   # 放置新食物
        else:
            self.snake.pop()     # 移除蛇尾（没吃到食物时蛇不生长）
        
        # 4. 检查游戏是否结束
        game_over = False
        if self.is_collision():
            print(f"fps {self.clock.get_fps()}")
            game_over = True
            self.draw_game_over()
            return game_over, self.score
        
        # 5. 更新界面和时钟
        self._update_ui()
        self.clock.tick(self.speed)
        
        # 6. 返回游戏状态
        return game_over, self.score
    
    def draw_game_over(self):
        """
        绘制游戏结束画面
        
        显示游戏结束信息和重新开始提示
        """
        font = pygame.font.Font(None, self.block_size)
        text = font.render('Game Over - Press Enter to Play Again', True, WHITE)
        text_rect = text.get_rect(center=(self.width/2, self.height/2))
        self.display.blit(text, text_rect)
        pygame.display.flip()
    
    def is_collision(self, point: Optional[Point] = None):
        """
        检查碰撞
        
        检查指定点（默认为蛇头）是否发生碰撞
        
        Args:
            point: 要检查的点，默认为蛇头
            
        Returns:
            是否发生碰撞
        """
        if point is None:
            point = self.head
            
        # 检查是否撞墙
        if (point.x > self.width - self.block_size or point.x < 0 or \
            point.y > self.height - self.block_size or point.y < 0):
            return True
            
        # 检查是否撞到自己
        for body in self.snake[1:]:
            if body.equal_with_block(point, self.block_size):
                return True
        return False
    
    def _update_ui(self):
        """
        更新游戏界面
        
        重绘整个游戏画面，包括背景、蛇身和食物
        """
        # 清空屏幕（黑色背景）
        self.display.fill(BLACK)
        
        # 绘制蛇身
        for pt in self.snake:
            # 外层蓝色方块
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            # 内层浅蓝色方块（创建立体效果）
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+(self.block_size // 5), pt.y+(self.block_size // 5), self.block_size * 3 / 5, self.block_size * 3 / 5))
        
        # 绘制食物（红色方块）
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))
        
        # 更新显示
        pygame.display.flip()
    
    def _move(self, direction):
        """
        根据方向移动蛇头
        
        Args:
            direction: 移动方向
        """
        x = self.head.x
        y = self.head.y
        
        # 根据方向更新坐标
        if direction == Direction.RIGHT:
            x += self.block_size
        elif direction == Direction.LEFT:
            x -= self.block_size
        elif direction == Direction.DOWN:
            y += self.block_size
        elif direction == Direction.UP:
            y -= self.block_size
            
        # 更新蛇头位置
        self.head = Point(x, y)

if __name__ == '__main__':
    """
    主程序入口
    
    创建游戏实例并运行主游戏循环
    """
    # 创建小尺寸的游戏用于测试
    game = SnakeGame(width=60, height=60, speed=10, block_size=4)
    
    # 主游戏循环
    while True:
        game_over, score = game.play_step()
        
        # 处理游戏结束
        if game_over:            
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    # 按回车键重新开始
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        game.reset()
                        waiting = False
    
    print(f'Final Score: {score}')
    pygame.quit()
