"""
工具函数模块

提供项目中使用的通用工具函数和类，包括：
1. 配置对象实例化功能
2. 便捷的字典访问类
3. 配置错误处理

作者：Snake Diffusion项目组
"""

from typing import Any, Union, Dict
import importlib

class ConfigurationError(Exception):
    """
    配置错误异常类
    
    当配置文件解析或对象实例化失败时抛出此异常
    """
    pass

def instantiate_from_config(config: Union[Dict[str, Any], Any]) -> Any:
    """
    从配置字典递归实例化对象
    
    这个函数可以根据配置字典中的__type__字段动态导入并实例化类。
    支持嵌套配置的递归处理，常用于从YAML配置文件创建复杂对象。
    
    Args:
        config: 包含__type__和参数的字典，或原始值
               __type__格式：'module.path.ClassName'
    
    Returns:
        实例化的对象，所有嵌套对象也会被递归实例化
        
    Raises:
        ConfigurationError: 当导入或实例化失败时
        
    Example:
        config = {
            '__type__': 'torch.nn.Linear',
            'in_features': 10,
            'out_features': 5
        }
        layer = instantiate_from_config(config)  # 创建nn.Linear(10, 5)
    """
    # 如果配置不是字典，直接返回（处理原始值）
    if not isinstance(config, dict):
        return config
    
    # 如果没有__type__字段，递归处理字典值
    if '__type__' not in config:
        return {
            key: instantiate_from_config(value)
            for key, value in config.items()
        }
    
    # 获取类型路径并从配置中移除
    type_path = config.pop('__type__')
    
    # 递归处理所有嵌套配置
    processed_config = {
        key: instantiate_from_config(value)
        for key, value in config.items()
    }
    
    try:
        # 导入类
        module_path, class_name = type_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        target_class = getattr(module, class_name)
        
        # 使用处理后的配置创建实例
        instance = target_class(**processed_config)
        return instance
    
    except (ImportError, AttributeError) as e:
        raise ConfigurationError(f"导入失败 {type_path}: {str(e)}")
    except Exception as e:
        raise ConfigurationError(f"实例化失败 {type_path}: {str(e)}")

class EasyDict(dict):
    """
    便捷字典类
    
    继承自dict，但允许使用属性语法访问字典元素。
    这使得访问配置参数更加方便和直观。
    
    Example:
        config = EasyDict({'learning_rate': 0.001, 'batch_size': 32})
        print(config.learning_rate)  # 0.001，等价于config['learning_rate']
        config.epochs = 100          # 等价于config['epochs'] = 100
    """

    def __getattr__(self, name: str) -> Any:
        """
        通过属性语法获取字典值
        
        Args:
            name: 属性名（字典键）
            
        Returns:
            对应的字典值
            
        Raises:
            AttributeError: 当键不存在时
        """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        通过属性语法设置字典值
        
        Args:
            name: 属性名（字典键）
            value: 要设置的值
        """
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """
        通过属性语法删除字典项
        
        Args:
            name: 属性名（字典键）
        """
        del self[name]