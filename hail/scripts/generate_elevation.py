#!/usr/bin/env python3
"""
生成 256x256 海拔栅格 elevation.npy

说明：
- 读取 Excel（默认：项目根目录下 data/location.xlsx，或根目录 location.xlsx）
- 复用 test3.py 中的研究区经纬度边界，将站点经纬度映射到 256x256 像素网格
- 使用 IDW（反距离加权）在整个网格插值海拔值
- 对海拔进行 min-max 归一化到 [0,1]，保存为 float32 的 numpy 文件

运行：
    python scripts/generate_elevation.py 
    # 可选参数： --excel /path/to/location.xlsx --out /path/to/elevation.npy

输出：
- 原始海拔（米）与归一化的统计信息（min/max）
- 保存到 data/elevation.npy（默认）
"""

import os
import argparse
from typing import Optional
import numpy as np
import pandas as pd


# 复用 test3.py 的经纬度边界（保持一致性）
LAT_MIN = 34.80708708708709
LAT_MAX = 38.41069069069069
LON_MIN = 107.24255771801371
LON_MAX = 111.73299784667067

GRID_SIZE = 256


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_excel_path(cli_path: Optional[str]) -> str:
    if cli_path and os.path.isfile(cli_path):
        return cli_path
    root = project_root()
    candidates = [
        os.path.join(root, "data", "location.xlsx"),
        os.path.join(root, "location.xlsx"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "未找到 Excel 文件。请将 location.xlsx 放在项目根目录或 data/ 下，"
        "或通过 --excel 指定路径。"
    )


def resolve_out_path(cli_path: Optional[str]) -> str:
    if cli_path:
        out_dir = os.path.dirname(cli_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        return cli_path
    root = project_root()
    out_dir = os.path.join(root, "data")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "elevation.npy")


def find_column(df: pd.DataFrame, keywords: list[str]) -> str:
    """在 DataFrame 列名中查找匹配关键字的列名（包含/不区分大小写）。"""
    cols = [str(c).strip() for c in df.columns]
    lower = [c.lower() for c in cols]
    for k in keywords:
        k_low = str(k).lower()
        for orig, low in zip(cols, lower):
            if k_low == low or k_low in low:
                return orig
    raise KeyError(f"无法在表头中找到列：{keywords}")


def load_station_data(xlsx_path: str) -> pd.DataFrame:
    """读取 Excel 并鲁棒识别表头行（如位于第2行），返回清洗后的站点数据。"""
    # 先不设表头读取，方便定位真实表头行
    raw = pd.read_excel(xlsx_path, header=None)
    # 在前若干行中寻找包含关键列名的行作为表头
    candidate_header_row = None
    header_keywords_groups = [
        ["经度", "longitude", "lon"],
        ["纬度", "latitude", "lat"],
        ["海拔", "海拔高度", "elevation", "alt", "altitude"],
    ]

    max_check_rows = min(10, len(raw))
    for ridx in range(max_check_rows):
        row_vals = [str(v).strip().lower() for v in list(raw.iloc[ridx].values)]
        if not row_vals:
            continue
        group_hits = 0
        for kws in header_keywords_groups:
            hit = any(any(k in rv for rv in row_vals) for k in kws)
            if hit:
                group_hits += 1
        if group_hits >= 2:  # 至少命中两组关键列名
            candidate_header_row = ridx
            break

    if candidate_header_row is None:
        # 回退：直接按第一行作表头
        df = pd.read_excel(xlsx_path)
    else:
        # 用识别的表头行重建 DataFrame
        columns = list(raw.iloc[candidate_header_row].values)
        df = raw.iloc[candidate_header_row + 1:].copy()
        df.columns = columns

    # 尝试匹配中文或英文列名
    try:
        lon_col = find_column(df, ["经度", "Longitude", "longitude", "lon", "LONG", "LON"])
        lat_col = find_column(df, ["纬度", "Latitude", "latitude", "lat", "LAT"])
        ele_col = find_column(df, ["海拔", "海拔高度", "Elevation", "elevation", "elev", "ALT", "alt", "altitude"])
    except KeyError as e:
        # 打印调试信息，帮助定位列名问题
        raise KeyError(
            f"列匹配失败：{e}. 识别到的列有：{list(df.columns)}。请确认 Excel 的经纬度/海拔列名。"
        )

    # 数值化并清洗
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    ele = pd.to_numeric(df[ele_col], errors="coerce")

    clean = pd.DataFrame({"lon": lon, "lat": lat, "ele": ele}).dropna()
    # 边界过滤
    in_bounds = (
        (clean["lat"] >= LAT_MIN) & (clean["lat"] <= LAT_MAX) &
        (clean["lon"] >= LON_MIN) & (clean["lon"] <= LON_MAX)
    )
    clean = clean[in_bounds]

    if clean.empty:
        raise ValueError("清洗后无有效站点数据，请检查坐标范围或列内容。")

    return clean.reset_index(drop=True)


def ll2pix(lat: np.ndarray, lon: np.ndarray):
    """经纬度映射到 256x256 像素网格坐标（与 test3.py 一致）。
    返回 (row_idx, col_idx) = (lat_pos, lon_pos)
    """
    lat_pos = ((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_SIZE).clip(0, GRID_SIZE - 1).astype(int)
    lon_pos = ((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_SIZE).clip(0, GRID_SIZE - 1).astype(int)
    return lat_pos, lon_pos


def idw_interpolate(px: np.ndarray, py: np.ndarray, values: np.ndarray, grid_h: int, grid_w: int, power: float = 2.0, eps: float = 1e-6) -> np.ndarray:
    """简单的 IDW 插值（分块计算，避免一次性占用过多内存）。"""
    grid = np.empty((grid_h, grid_w), dtype=np.float64)
    # 坐标网格
    xs = np.arange(grid_h, dtype=np.float64)
    ys = np.arange(grid_w, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")  # X: row/lat, Y: col/lon

    # 对于与站点完全重合的像素，直接赋值
    grid[:] = np.nan
    grid[px, py] = values

    # 分块插值
    block = 32
    n = len(values)
    for r0 in range(0, grid_h, block):
        r1 = min(r0 + block, grid_h)
        # 当前块的坐标
        Xb = X[r0:r1, :]
        Yb = Y[r0:r1, :]
        # 与所有站点的距离
        dx = Xb[..., None] - px[None, None, :]
        dy = Yb[..., None] - py[None, None, :]
        dist = np.sqrt(dx * dx + dy * dy)

        # 权重（避免除零）
        w = 1.0 / np.power(dist + eps, power)
        # 将重合点的权重设为很大，确保精确复制站点值
        same = (dist < eps)
        if np.any(same):
            # 对于重合像素，直接赋值并跳过插值计算
            # 但保持权重计算的通用性
            pass

        # 加权平均
        num = (w * values[None, None, :]).sum(axis=-1)
        den = w.sum(axis=-1)
        interp = num / np.maximum(den, eps)

        # 写入块（保留已直接赋值的像素）
        blk = grid[r0:r1, :]
        mask = np.isnan(blk)
        blk[mask] = interp[mask]
        grid[r0:r1, :] = blk

    return grid


def normalize01(arr: np.ndarray) -> np.ndarray:
    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))
    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        raise ValueError(f"归一化失败，范围异常：min={amin}, max={amax}")
    out = (arr - amin) / (amax - amin)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="生成 256x256 海拔栅格 elevation.npy")
    parser.add_argument("--excel", type=str, default=None, help="Excel 路径（默认 data/location.xlsx 或 location.xlsx）")
    parser.add_argument("--out", type=str, default=None, help="输出 .npy 路径（默认 data/elevation.npy）")
    args = parser.parse_args()

    xlsx_path = resolve_excel_path(args.excel)
    out_path = resolve_out_path(args.out)

    print(f"读取 Excel: {xlsx_path}")
    df = load_station_data(xlsx_path)

    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    ele = df["ele"].to_numpy()

    # 映射到像素网格
    px, py = ll2pix(lat, lon)

    # 插值到完整网格（米）
    grid_m = idw_interpolate(px, py, ele.astype(np.float64), GRID_SIZE, GRID_SIZE, power=2.0)

    # 归一化到 [0,1]
    grid_norm = normalize01(grid_m)

    # 保存
    np.save(out_path, grid_norm)

    print("生成完成：")
    print(f"  输出文件: {out_path}")
    print(f"  原始海拔(米)范围: min={float(np.nanmin(grid_m)):.3f}, max={float(np.nanmax(grid_m)):.3f}")
    print(f"  归一化范围: min={float(np.min(grid_norm)):.3f}, max={float(np.max(grid_norm)):.3f}")
    print(f"  shape={grid_norm.shape}, dtype={grid_norm.dtype}")


if __name__ == "__main__":
    main()