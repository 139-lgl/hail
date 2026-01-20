"""
每小时采样的冰雹数据集（支持单槽/多槽/宽松索引）

用途：
- 固定槽模式（默认）：每小时选择第 bucket_index 张（例如第 0 张），跨小时构造序列。
- 多槽模式：选择该小时内的所有时间槽（例如 10 个 6 分钟间隔），为每个槽构造跨小时序列，并将所有序列合并，显著增大样本量。

特点：
- 解析文件名中的时间戳（支持 YYYYMMDDHHMM 或至少 YYYYMMDDHH）。
- 先按小时分桶，对每小时内部按分钟稳定排序；分钟缺失时使用位置索引作为回退。
- 分配到时间槽：默认每小时 10 槽（每 6 分钟一个槽）；分钟缺失时回退按索引模 `slots_per_hour` 分组。
- 与小时级气象 .npy 文件按键 YYYYMMDDHH 对齐（每个小时一个动作文件）。
- 可选海拔通道，使用与现有 HailDataset 相同的下采样与归一化策略。

使用建议：
- 如果你希望“用每个小时的所有 6 分钟帧”，将 `bucket_index` 设为 -1（启用多槽模式），并保留默认 `slots_per_hour=10`。
- 如果某些小时缺帧，可将 `strict=False`（宽松模式），将选择最接近的可用槽以保持序列连续性。
"""

import os
from typing import List, Any, Tuple, Optional, Dict

import PIL.Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import PIL


def _extract_time_parts(filename: str) -> Optional[Tuple[str, Optional[int]]]:
    """
    从文件名提取小时键与分钟。

    返回 (hour_key, minute)；hour_key 为 YYYYMMDDHH，minute 为 [0,59] 或 None。
    文件名中若无足够数字，返回 None。
    """
    digits = "".join(ch for ch in filename if ch.isdigit())
    if len(digits) < 10:
        return None
    hour_key = digits[:10]
    minute = int(digits[10:12]) if len(digits) >= 12 else None
    return hour_key, minute


class HourlyHailDataset(Dataset):
    """
    每小时采样数据集（冰雹雷达 + 小时气象）

    - images_dir: 图像根目录，包含按天的子目录（如 visualization_2/20220511）
    - actions_path: 气象 .npy 根目录（如 visualization_2/final）
    - seq_length: 序列长度（历史帧数）
    - bucket_index: 每小时选择第几张（0 表示第一张，1 表示第二张）
    - transform: 图像变换
    - elevation_path: 可选海拔 .npy，形状 [H,W]，值域[0,1]
    - strict: 若某小时不足 bucket_index+1 张，strict=True 则跳过，False 则选择最接近的张次
    """

    def __init__(
        self,
        images_dir: str,
        actions_path: str,
        seq_length: int,
        bucket_index: int = 0,
        transform: Optional[Any] = None,
        elevation_path: Optional[str] = None,
        strict: bool = True,
        slots_per_hour: int = 10,
    ) -> None:
        super().__init__()

        self.images_dir = images_dir
        self.actions_path = actions_path
        self.seq_length = seq_length
        self.bucket_index = bucket_index
        self.transform = transform
        self.elevation_path = elevation_path
        self.strict = strict
        self.slots_per_hour = max(1, int(slots_per_hour))

        # 与 HailDataset 保持一致的下采样策略（256x256 -> 64x64 或 64x64 -> 16x16）
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=4, padding=0)

        # 预处理海拔通道
        self.elev_downsample: Optional[torch.Tensor] = None
        if self.elevation_path and os.path.exists(self.elevation_path):
            try:
                elev = np.load(self.elevation_path)
                if elev.ndim == 2:
                    with torch.no_grad():
                        t = torch.from_numpy(elev).float()
                        t = t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                        t = self.conv(t).squeeze(0).squeeze(0)  # [Hs,Ws]
                        self.elev_downsample = t
                else:
                    self.elev_downsample = None
            except Exception:
                self.elev_downsample = None

        # hour_key -> [(minute, rel_img_path)]
        hours_map: Dict[str, List[Tuple[Optional[int], str]]] = {}

        # 扫描天级子目录并收集 PNG 图像
        all_dirs = sorted([d for d in os.listdir(self.images_dir)])
        for d in all_dirs:
            # 跳过可能的动作输出目录
            if d.lower() in {"final", "final_2"}:
                continue
            dir_path = os.path.join(self.images_dir, d)
            if not os.path.isdir(dir_path):
                continue

            for item in os.listdir(dir_path):
                if not item.endswith(".png"):
                    continue
                parts = _extract_time_parts(item)
                if parts is None:
                    # 文件名不含足够的时间数字，跳过
                    continue
                hour_key, minute = parts
                rel_path = f"{d}/{item}"
                hours_map.setdefault(hour_key, []).append((minute, rel_path))

        # 每小时内排序（分钟优先；分钟缺失的项排在后面）
        for hk in hours_map.keys():
            hours_map[hk].sort(key=lambda x: (x[0] if x[0] is not None else 1_000_000))

        # 基于小时键排序
        hour_keys_sorted = sorted(hours_map.keys(), key=lambda s: int(s))

        # 将每小时的帧分配到槽（默认 10 槽：每 6 分钟一个槽）
        # 分配策略：
        # - 若有分钟：slot_idx = minute // slot_width
        # - 若分钟缺失：slot_idx = position_index % slots_per_hour 作为回退
        slot_width = max(1, 60 // self.slots_per_hour)

        # 构造每个小时的槽映射：hk -> {slot_idx: [rel_img_path, ...]}
        hours_slot_map: Dict[str, Dict[int, List[str]]] = {}
        for hk in hour_keys_sorted:
            lst = hours_map[hk]
            hours_slot_map[hk] = {}
            for pos, (minute, rel) in enumerate(lst):
                if minute is not None:
                    slot_idx = int(minute // slot_width)
                    # 边界保护
                    if slot_idx >= self.slots_per_hour:
                        slot_idx = self.slots_per_hour - 1
                else:
                    # 分钟缺失时的回退：按索引模 slots_per_hour 归槽
                    slot_idx = pos % self.slots_per_hour
                hours_slot_map[hk].setdefault(slot_idx, []).append(rel)

        def _pick_for_slot(hk: str, target_slot: int) -> Optional[str]:
            """从该小时的指定槽中选一张；若该槽缺失则按 strict/loose 决策。"""
            slot_map = hours_slot_map.get(hk, {})
            if target_slot in slot_map:
                return slot_map[target_slot][0]
            if self.strict:
                return None
            # 宽松模式：选择距离 target_slot 最近的可用槽
            if not slot_map:
                return None
            nearest_slot = min(slot_map.keys(), key=lambda s: abs(s - target_slot))
            return slot_map[nearest_slot][0]

        # 支持两种模式：
        # 1) 固定槽（bucket_index >= 0）
        # 2) 多槽模式（bucket_index < 0）：为每个槽都构建跨小时序列并合并
        self.sequences: List[Tuple[List[str], List[str]]] = []

        if self.bucket_index >= 0:
            picked_imgs: List[str] = []
            picked_actions: List[str] = []
            for hk in hour_keys_sorted:
                rel = _pick_for_slot(hk, self.bucket_index)
                if rel is None:
                    continue
                picked_imgs.append(rel)
                picked_actions.append(f"{hk}.npy")

            # 构建滑窗序列
            for i in range(self.seq_length + 1, len(picked_imgs)):
                img_sequence = picked_imgs[max(i - self.seq_length - 1, 0): i]
                action_sequence = picked_actions[max(i - self.seq_length, 0): i]
                self.sequences.append((img_sequence, action_sequence))
        else:
            # 多槽模式：为每个槽（0..slots_per_hour-1）分别构造序列，并合并
            for slot in range(self.slots_per_hour):
                picked_imgs_s: List[str] = []
                picked_actions_s: List[str] = []
                for hk in hour_keys_sorted:
                    rel = _pick_for_slot(hk, slot)
                    if rel is None:
                        continue
                    picked_imgs_s.append(rel)
                    picked_actions_s.append(f"{hk}.npy")

                # 每个槽生成自己的滑窗序列后合并到总列表
                for i in range(self.seq_length + 1, len(picked_imgs_s)):
                    img_sequence = picked_imgs_s[max(i - self.seq_length - 1, 0): i]
                    action_sequence = picked_actions_s[max(i - self.seq_length, 0): i]
                    self.sequences.append((img_sequence, action_sequence))

        # 行动通道数：风向、风速 + （可选）海拔
        self.actions_channels = 2 + (1 if self.elev_downsample is not None else 0)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs, actions = self.sequences[index]
        last_img_rel = imgs[-1]

        def _load_img(rel_path: str) -> Any:
            img_path = os.path.join(self.images_dir, rel_path)
            image = PIL.Image.open(img_path).convert("RGB")
            if self.transform is not None:
                return self.transform(image)
            else:
                return image

        # 加载动作序列并下采样/归一化
        actions_data: List[torch.Tensor] = []
        for ac in actions:
            temp = np.load(os.path.join(self.actions_path, ac))
            # 归一化到 [0,1]
            temp[0] = temp[0] / 360.0
            temp[1] = temp[1] / 360.0
            with torch.no_grad():
                t = torch.from_numpy(temp).float()  # [2,H,W]
                t = t.unsqueeze(1)                 # [2,1,H,W]，批次大小=2，通道=1
                t = self.conv(t).squeeze(1)        # [2,Hs,Ws]
                if self.elev_downsample is not None:
                    t = torch.cat([t, self.elev_downsample.unsqueeze(0)], dim=0)  # [2/3,Hs,Ws]
            actions_data.append(t)

        return (
            _load_img(last_img_rel),
            torch.stack([_load_img(p) for p in imgs[:-1]]),
            torch.stack(actions_data, dim=0)
        )
