"""
数据集处理模块

这个模块定义了用于训练扩散模型的数据集类，具有以下特点：
1. SequencesDataset：处理贪吃蛇游戏的序列数据
2. HailDataset：处理冰雹天气数据的序列数据
3. 支持图像序列和动作序列的同步加载
4. 提供数据预处理和变换功能

主要功能：
- 序列数据加载：按时间顺序加载图像和动作序列
- 数据预处理：图像变换、动作归一化等
- 批量处理：支持PyTorch DataLoader的批量训练
- 多数据源：支持不同格式的数据集

作者：Snake Diffusion项目组
"""

import os
from typing import List, Any, Tuple, Optional

import PIL.Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
import os

def _dbg_print(msg: str):
    if os.environ.get("DEBUG_STEP") == "1":
        print(msg)

class SequencesDataset(Dataset):
    """
    序列数据集类
    
    用于加载贪吃蛇游戏的图像序列和动作序列数据，支持：
    - 按时间顺序的图像序列加载
    - 对应的动作序列加载
    - 可配置的序列长度
    - 图像预处理变换
    
    数据格式：
    - 图像：按编号命名的JPG文件（0.jpg, 1.jpg, ...）
    - 动作：文本文件，每行一个动作编号
    """
    
    def __init__(
        self,
        images_dir: str,
        actions_path: str,
        seq_length: int,
        transform: Optional[Any] = None
    ) -> None:
        """
        初始化序列数据集
        
        Args:
            images_dir: 图像文件夹路径
            actions_path: 动作文件路径
            seq_length: 序列长度
            transform: 图像变换函数（可选）
        """
        super().__init__()
        
        # 获取并排序图像文件路径
        paths = sorted(
            [ item for item in os.listdir(images_dir) if item.endswith(".jpg")],
            key=lambda item: int(item.split(".")[0])  # 按文件名中的数字排序
        )
        self.images_dir = images_dir
        
        # 读取动作序列
        with open(actions_path) as file:
            actions = [int(line) for line in file.readlines()]
            
        # 确保图像和动作数量一致
        assert len(actions) == len(paths)
        
        # 构建序列数据
        self.sequences: List[Tuple[List[str], List[int]]] = []
        self.transform = transform
        
        # 为每个可能的序列创建数据项
        for i in range(seq_length + 1, len(paths)):
            # 获取序列图像路径和对应动作
            img_sequence = paths[max(i-seq_length - 1, 0) : i]
            action_sequence = actions[max(i - seq_length, 0) : i]
            self.sequences.append((img_sequence, action_sequence))

    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            序列数量
        """
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取指定索引的数据项
        
        Args:
            index: 数据项索引
            
        Returns:
            tuple: (目标图像, 历史图像序列, 动作序列)
                - 目标图像: 序列的最后一帧 [C, H, W]
                - 历史图像序列: 前面的帧 [seq_len, C, H, W]
                - 动作序列: 对应的动作 [seq_len]
        """
        imgs, actions = self.sequences[index]
        last_img = imgs[-1]  # 目标图像（序列最后一帧）
        
        def do_transform(img_path: str) -> Any:
            """
            加载并变换图像
            
            Args:
                img_path: 图像文件名
                
            Returns:
                变换后的图像张量
            """
            img_path = os.path.join(self.images_dir, img_path)
            image = PIL.Image.open(img_path).convert('RGB')
            if self.transform is not None:
                return self.transform(image)
            else:
                return image
                
        return (
            do_transform(last_img),                                    # 目标图像
            torch.stack([do_transform(img) for img in imgs[:-1]]),     # 历史图像序列
            torch.tensor(actions)                                      # 动作序列
        )

class HailDataset(Dataset):
    """
    冰雹数据集类
    
    用于加载冰雹天气数据的图像序列和气象参数序列，支持：
    - 多目录的数据组织结构
    - 气象参数的归一化处理
    - 图像下采样处理
    - 复杂的数据预处理流程
    
    数据格式：
    - 图像：PNG格式，按目录和编号组织
    - 气象参数：NPY格式，包含风向和风速信息
    """
    
    def __init__(
        self,
        images_dir: str,
        actions_path: str,
        seq_length: int,
        transform: Optional[Any] = None,
        elevation_path: Optional[str] = None
    ) -> None:
        """
        初始化冰雹数据集
        
        Args:
            images_dir: 图像根目录路径
            actions_path: 气象参数根目录路径
            seq_length: 序列长度
            transform: 图像变换函数（可选）
        """
        super().__init__()
        paths = []
        actions = []
        
        # 获取所有子目录
        allDir = os.listdir(images_dir)
        allDir = sorted(allDir)
        
        self.images_dir = images_dir
        self.sequences: List[Tuple[List[str], List[int]]] = []
        self.transform = transform
        self.actions_path = actions_path
        self.elevation_path = elevation_path
        
        # 用于气象参数下采样的卷积层（256x256 -> 64x64，或64x64 -> 16x16）
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=4, padding=0)

        # 可选：加载并预处理海拔数据，作为额外通道（静态条件）
        self.elev_downsample: Optional[torch.Tensor] = None
        if self.elevation_path and os.path.exists(self.elevation_path):
            try:
                elev = np.load(self.elevation_path)
                # 期望形状为 [H, W]，值域已归一化到[0,1]
                if elev.ndim == 2:
                    with torch.no_grad():
                        t = torch.from_numpy(elev).float()
                        # 下采样到与动作一致的分辨率（使用相同的卷积策略）
                        t = t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                        t = self.conv(t).squeeze(0).squeeze(0)  # [64,64] 或 [16,16]
                        self.elev_downsample = t
                else:
                    # 如果维度不符，忽略海拔通道
                    self.elev_downsample = None
            except Exception:
                self.elev_downsample = None

        # 行动通道数：风向、风速 + （可选）海拔
        self.actions_channels = 2 + (1 if self.elev_downsample is not None else 0)
        
        # 遍历每个数据目录
        for path in allDir:
            if(path == "final"):  # 跳过特殊目录
                continue
                
            # 获取当前目录下的图像文件
            paths.append(sorted(
                [ path + "/" + item for item in os.listdir(images_dir + "/" + path) if item.endswith(".png")],
                key=lambda item: int(item.split("/")[1].split(".")[0])  # 按文件名数字排序
            ))
            
            # 构建对应的气象参数文件名
            temp_actions = []
            for cur in paths[-1]:
                # 从图像文件名生成对应的.npy文件名
                temp_actions.append(cur.split("/")[1][:10]+".npy")
            actions.append(temp_actions)
            
            # 确保图像和气象参数数量一致
            assert len(actions[-1]) == len(paths[-1])
            
            # 为当前目录创建序列数据
            for i in range(seq_length + 1, len(paths[-1])):
                img_sequence = paths[-1][max(i-seq_length - 1, 0) : i]
                action_sequence = actions[-1][max(i - seq_length, 0) : i]
                self.sequences.append((img_sequence, action_sequence))

    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            序列数量
        """
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取指定索引的数据项
        
        Args:
            index: 数据项索引
            
        Returns:
            tuple: (目标图像, 历史图像序列, 气象参数序列)
                - 目标图像: 序列的最后一帧 [C, H, W]
                - 历史图像序列: 前面的帧 [seq_len, C, H, W]
                - 气象参数序列: 处理后的气象数据 [seq_len, 16, 16]
        """
        imgs, actions = self.sequences[index]
        last_img = imgs[-1]  # 目标图像
        
        def do_transform(img_path: str) -> Any:
            """
            加载并变换图像
            
            Args:
                img_path: 图像文件路径
                
            Returns:
                变换后的图像张量
            """
            img_path = os.path.join(self.images_dir, img_path)
            image = PIL.Image.open(img_path).convert('RGB')
            if self.transform is not None:
                return self.transform(image)
            else:
                return image
        
        # 处理气象参数序列
        # 下采样到64x64分辨率（若原始是256x256）
        # 风向和风速归一化到[0,1]范围
        actions_data = []
        for ac in actions:
            # 加载气象参数数据
            temp = np.load(self.actions_path + "/" + ac)
            temp[0] = temp[0] / 360.0  # 风向归一化（0-360度 -> 0-1）
            temp[1] = temp[1] / 360.0  # 风速归一化（假设最大360）
            
            # 转换为张量并下采样
            with torch.no_grad():
                temp = torch.from_numpy(temp).float()
                temp = temp.unsqueeze(1)  # 添加通道维度
                temp = self.conv(temp).squeeze(1)  # 4x4下采样

                # 并入海拔额外通道（静态条件）
                if self.elev_downsample is not None:
                    # temp: [2, Hs, Ws], elev_downsample: [Hs, Ws]
                    temp = torch.cat([temp, self.elev_downsample.unsqueeze(0)], dim=0)  # [2+1, Hs, Ws]
            actions_data.append(temp)
        
        return (
            do_transform(last_img),                                    # 目标图像
            torch.stack([do_transform(img) for img in imgs[:-1]]),     # 历史图像序列
            torch.stack(actions_data, dim=0)                           # 气象参数序列（形状：[seq_len, C(2或3), Hs, Ws]）
        )

class TiledHailDataset(Dataset):
    def __init__(
        self,
        images_root: str,
        wind_root: str,
        elev_root: Optional[str],
        seq_length: int,
        transform: Optional[Any] = None,
        tile_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.images_root = images_root
        self.wind_root = wind_root
        self.elev_root = elev_root
        self.seq_length = seq_length
        self.transform = transform
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=4, padding=0)
        if tile_indices is None:
            tile_indices = list(range(16))
        self.tile_indices = [int(i) for i in tile_indices]
        self.elev_downsample_map = {}
        if self.elev_root and os.path.isdir(self.elev_root):
            for ti in self.tile_indices:
                p = os.path.join(self.elev_root, f"{ti}.npy")
                if os.path.isfile(p):
                    try:
                        arr = np.load(p)
                        if arr.ndim == 2:
                            with torch.no_grad():
                                t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
                                t = self.conv(t).squeeze(0).squeeze(0)
                                self.elev_downsample_map[ti] = t
                    except Exception:
                        pass
        self.sequences: List[Tuple[List[str], List[str]]] = []
        days = sorted([d for d in os.listdir(self.images_root) if os.path.isdir(os.path.join(self.images_root, d))])
        for day in days:
            img_day_dir = os.path.join(self.images_root, day, "image")
            if not os.path.isdir(img_day_dir):
                continue
            for ti in self.tile_indices:
                tile_dir = os.path.join(img_day_dir, str(ti))
                if not os.path.isdir(tile_dir):
                    continue
                names = sorted([f for f in os.listdir(tile_dir) if f.endswith(".png")])
                valid_imgs: List[str] = []
                valid_actions: List[str] = []
                for n in names:
                    tkey = n.split("_")[0]
                    wind_rel = os.path.join(day, str(ti), f"{tkey}_{ti}.npy")
                    wind_fp = os.path.join(self.wind_root, wind_rel)
                    if not os.path.isfile(wind_fp):
                        continue
                    valid_imgs.append(os.path.join(day, "image", str(ti), n))
                    valid_actions.append(wind_rel)
                for i in range(self.seq_length + 1, len(valid_imgs)):
                    img_sequence = valid_imgs[max(i - self.seq_length - 1, 0): i]
                    action_sequence = valid_actions[max(i - self.seq_length, 0): i]
                    self.sequences.append((img_sequence, action_sequence))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs, actions = self.sequences[index]
        last_img_rel = imgs[-1]
        def _load_img(rel_path: str) -> Any:
            img_path = os.path.join(self.images_root, rel_path)
            image = PIL.Image.open(img_path).convert("RGB")
            if self.transform is not None:
                return self.transform(image)
            else:
                return image
        day = last_img_rel.split("/")[0]
        try:
            center_ti = int(last_img_rel.split("/")[2])
        except Exception:
            center_ti = None
        def _neighbors(ti: Optional[int]) -> List[Optional[int]]:
            if ti is None:
                return [None] * 9
            r = ti // 4
            c = ti % 4
            offsets = [(0,0), (-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            res: List[Optional[int]] = []
            for dr, dc in offsets:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < 4 and 0 <= nc < 4:
                    res.append(nr * 4 + nc)
                else:
                    res.append(None)
            return res
        neigh_ids = _neighbors(center_ti)
        def _find_neighbor_img(day: str, nb_ti: int, tkey: str) -> Optional[str]:
            nb_dir = os.path.join(self.images_root, day, "image", str(nb_ti))
            if not os.path.isdir(nb_dir):
                return None
            try:
                names = [f for f in os.listdir(nb_dir) if f.endswith(".png") and f.startswith(tkey)]
            except Exception:
                names = []
            if not names:
                return None
            name = sorted(names)[0]
            return os.path.join(day, "image", str(nb_ti), name)
        prev_imgs_steps: List[torch.Tensor] = []
        actions_steps: List[torch.Tensor] = []
        for ac_rel, img_rel in zip(actions, imgs[:-1]):
            tkey = ac_rel.split("/")[-1].split("_")[0]
            try:
                center_ti_cur = int(ac_rel.split("/")[1])
            except Exception:
                center_ti_cur = center_ti
            cur_neigh_ids = _neighbors(center_ti_cur)
            step_imgs: List[torch.Tensor] = []
            step_actions: List[torch.Tensor] = []
            center_arr = np.load(os.path.join(self.wind_root, ac_rel))
            with torch.no_grad():
                t_center = torch.from_numpy(center_arr).float().unsqueeze(1)
                t_center = self.conv(t_center).squeeze(1)
                Hs, Ws = t_center.shape[-2], t_center.shape[-1]
            use_elev = len(self.elev_downsample_map) > 0
            for idx, nb_ti in enumerate(cur_neigh_ids):
                if nb_ti is not None:
                    nb_wind_rel = os.path.join(day, str(nb_ti), f"{tkey}_{nb_ti}.npy")
                    nb_wind_fp = os.path.join(self.wind_root, nb_wind_rel)
                    if os.path.isfile(nb_wind_fp):
                        arr = np.load(nb_wind_fp)
                        with torch.no_grad():
                            t = torch.from_numpy(arr).float().unsqueeze(1)
                            t = self.conv(t).squeeze(1)
                            if use_elev:
                                if nb_ti in self.elev_downsample_map:
                                    t = torch.cat([t, self.elev_downsample_map[nb_ti].unsqueeze(0)], dim=0)
                                else:
                                    t = torch.cat([t, torch.zeros((1, Hs, Ws), dtype=torch.float32)], dim=0)
                        step_actions.append(t)
                    else:
                        with torch.no_grad():
                            t = torch.zeros((2, Hs, Ws), dtype=torch.float32)
                            if use_elev:
                                if nb_ti in self.elev_downsample_map:
                                    t = torch.cat([t, self.elev_downsample_map[nb_ti].unsqueeze(0)], dim=0)
                                else:
                                    t = torch.cat([t, torch.zeros((1, Hs, Ws), dtype=torch.float32)], dim=0)
                        step_actions.append(t)
                    nb_img_rel = _find_neighbor_img(day, nb_ti, tkey)
                    if nb_img_rel is not None:
                        step_imgs.append(_load_img(nb_img_rel))
                    else:
                        step_imgs.append(_load_img(img_rel))
                else:
                    with torch.no_grad():
                        t = torch.zeros((2, Hs, Ws), dtype=torch.float32)
                        if use_elev:
                            t = torch.cat([t, torch.zeros((1, Hs, Ws), dtype=torch.float32)], dim=0)
                    step_actions.append(t)
                    step_imgs.append(_load_img(img_rel))
            actions_steps.append(torch.stack(step_actions, dim=0))
            step_imgs_tensor = torch.stack(step_imgs, dim=0)
            _dbg_print(f"dataset_step_imgs {tuple(step_imgs_tensor.shape)}")
            _dbg_print(f"dataset_step_actions {tuple(actions_steps[-1].shape)}")
            prev_imgs_steps.append(step_imgs_tensor)
        target_img = _load_img(last_img_rel)
        prev_imgs = torch.stack(prev_imgs_steps, dim=0)
        actions_tensor = torch.stack(actions_steps, dim=0)
        _dbg_print(f"dataset_prev_imgs {tuple(prev_imgs.shape)}")
        _dbg_print(f"dataset_actions_tensor {tuple(actions_tensor.shape)}")
        return (target_img, prev_imgs, actions_tensor)

if __name__ == "__main__":
    """
    测试代码
    
    创建数据集实例并测试基本功能
    """
    import torchvision.transforms as transforms
    
    # 测试SequencesDataset
    dataset = SequencesDataset(
        images_dir="training_data_directions/snapshots",
        actions_path="training_data_directions/actions",
        seq_length=3,
        transform=transforms.ToTensor()
    )
    dataset[0]
