"""
数据集生成脚本

这个脚本用于训练Q学习智能体并生成训练数据集，具有以下功能：
1. 加载配置文件和环境设置
2. 初始化Q学习智能体
3. 训练智能体并收集游戏数据
4. 保存训练好的模型和数据集

主要特点：
- 支持命令行参数配置
- 可选择是否记录游戏过程
- 支持从检查点恢复训练
- 可视化训练过程（可选）
- 自动清理数据集目录

使用方法：
    python generate_dataset.py --config config/SnakeAgent.yaml --model model_path --dataset dataset_path

作者：Snake Diffusion项目组
"""

import click
import yaml

from utils.utils import EasyDict, instantiate_from_config
from q_agent import QAgent, QAgentConfig
from game.env import Environment

@click.command()
@click.option('--config', help='配置文件路径（YAML格式）', metavar='YAML', type=str, required=True, default="config/SnakeAgent.yaml")
@click.option('--model', help='模型保存路径', type=str, required=True)
@click.option('--dataset', help='数据集保存路径', type=str, required=True)
@click.option('--record', help='是否记录动作和游戏快照', is_flag=True)
@click.option('--clear-dataset', help='是否清空数据集文件夹', is_flag=True, required=False, default=False)
@click.option('--show-plot', help='是否显示训练过程图表', is_flag=True, required=False, default=False)
@click.option('--last-checkpoint', help='恢复训练的检查点路径', type=str, required=False)
def main(**kwargs):
    """
    主函数：数据集生成和Q学习智能体训练
    
    这个函数执行以下步骤：
    1. 解析命令行参数
    2. 加载配置文件
    3. 初始化游戏环境
    4. 创建Q学习智能体
    5. 开始训练并生成数据集
    
    Args:
        **kwargs: 命令行参数字典，包含：
            - config: 配置文件路径
            - model: 模型保存路径
            - dataset: 数据集保存路径
            - record: 是否记录游戏过程
            - clear_dataset: 是否清空数据集
            - show_plot: 是否显示训练图表
            - last_checkpoint: 检查点路径（可选）
    """
    # 将命令行参数转换为EasyDict对象，便于访问
    options = EasyDict(kwargs)
    
    # 加载YAML配置文件
    with open(options.config, 'r') as f:
        config = EasyDict(**yaml.safe_load(f))
    
    # 根据配置实例化游戏环境
    # 这里使用工厂模式，根据配置动态创建环境对象
    env: Environment = instantiate_from_config(config.env)
    
    # 创建Q学习智能体配置
    q_agent_config = QAgentConfig(**instantiate_from_config(config.q_agent))
    
    # 初始化Q学习智能体
    # 传入环境、配置、模型路径、数据集路径和可选的检查点路径
    q_agent = QAgent(
        env, 
        q_agent_config, 
        options.model, 
        options.dataset, 
        options.get("last_checkpoint", None)
    )
    
    # 开始训练过程
    # show_plot: 是否显示训练过程的可视化图表
    # record: 是否记录游戏状态和动作用于生成数据集
    # clear_dataset: 是否在开始前清空数据集目录
    q_agent.train(options.show_plot, options.record, options.clear_dataset)

if __name__ == "__main__":
    """
    脚本入口点
    
    当直接运行此脚本时，调用main函数处理命令行参数
    """
    main()