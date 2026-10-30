# 批量生成SPAD伪彩色图（8个样本）
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from pathlib import Path

# 设置基础目录
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'output' / 'models'
FIGURE_DIR = BASE_DIR / 'output' / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 实际存在的8个样本列表
SAMPLES = [
    'CK_HS-45', 'CK_HS-17', 'DS_HS-17', 'DS_HS-103',
    'DS_HS-99', 'DS_HS-45', 'DS_HS-58', 'DS_HS-116'
]
# 参数
IMG_SIZE = (480, 480)
CMAP_OPENCV = cv2.COLORMAP_TURBO
CMAP_SEABORN = 'jet'

# SPAD范围（训练集范围）
SPAD_MIN = 13.2
SPAD_MAX = 36.15

def generate_spad_map(sample_name):
    """为单个样本生成SPAD伪彩色图"""
    file_name = f"{sample_name}_maize_pixel_spectra_predictions_MA_D2"
    excel_path = MODEL_DIR / f"{file_name}.xlsx"
    
    if not excel_path.exists():
        print(f"[MISSING] {sample_name}: 预测文件不存在")
        return False
    
    # 输出文件路径
    out_png = FIGURE_DIR / f"{file_name}_SPAD_pseudo.png"
    out_seaborn = FIGURE_DIR / f"{file_name}_SPAD_seaborn.png"
    
    # 读取预测结果
    df = pd.read_excel(excel_path)
    
    # 创建空画布
    H, W = IMG_SIZE
    canvas = np.full((H, W), np.nan, dtype=float)
    
    # 填充数值
    for _, row in df.iterrows():
        y, x, val = int(row['y']), int(row['x']), row['Predicted_SPAD']
        if 0 <= y < H and 0 <= x < W:
            canvas[y, x] = val
    
    # 创建掩模（标记植株区域）
    mask = ~np.isnan(canvas)
    
    # 标记超出训练范围的像素
    out_of_range_low = mask & (canvas < SPAD_MIN)
    out_of_range_high = mask & (canvas > SPAD_MAX)
    in_range = mask & (canvas >= SPAD_MIN) & (canvas <= SPAD_MAX)
    
    # 归一化（仅对植株区域，限制在训练范围内）
    if np.any(in_range):
        canvas_normalized = np.zeros_like(canvas)
        canvas_normalized[in_range] = 255 * (canvas[in_range] - SPAD_MIN) / (SPAD_MAX - SPAD_MIN)
    else:
        canvas_normalized = np.zeros_like(canvas)
    
    # 转换为 uint8
    canvas_uint8 = np.clip(canvas_normalized, 0, 255).astype(np.uint8)
    
    # OpenCV 伪彩色图
    pseudo = cv2.applyColorMap(canvas_uint8, CMAP_OPENCV)
    
    # 标记超出范围的像素：蓝色(< 13.2)，红色(> 36.15)
    pseudo[out_of_range_low] = [255, 0, 0]  # BGR 蓝色
    pseudo[out_of_range_high] = [0, 0, 255]  # BGR 红色
    
    # 将非植株区域设置为黑色
    pseudo[~mask] = [0, 0, 0]  # BGR 黑色
    
    # 保存图像
    cv2.imwrite(str(out_png), pseudo)
    
    # Seaborn 预览图
    plt.figure(figsize=(5, 5))
    
    # 创建背景为黑色的画布
    background = np.zeros(canvas.shape)
    norm = Normalize(vmin=SPAD_MIN, vmax=SPAD_MAX)
    
    # 绘制黑色背景
    sns.heatmap(background, cmap='gray', cbar=False, square=True,
                vmin=0, vmax=0, alpha=1.0)
    
    # 创建用于显示的canvas，超出范围的值用NaN替换
    canvas_display = canvas.copy()
    canvas_display[out_of_range_low] = np.nan
    canvas_display[out_of_range_high] = np.nan
    
    # 显示正常范围内的热力图
    ax = sns.heatmap(canvas_display,
                cmap=CMAP_SEABORN,
                cbar_kws={
                    'fraction': 0.045,
                    'aspect': 20,
                    'pad': 0.05
                },
                square=True,
                linewidths=0,
                alpha=1.0,
                mask=~mask,
                norm=norm,
                xticklabels=False,
                yticklabels=False)
    
    # 标记超出范围的像素
    y_low, x_low = np.where(out_of_range_low)
    y_high, x_high = np.where(out_of_range_high)
    
    if len(x_low) > 0:
        ax.scatter(x_low + 0.5, y_low + 0.5, c='blue', s=1, marker='s', label=f'< {SPAD_MIN}')
    if len(x_high) > 0:
        ax.scatter(x_high + 0.5, y_high + 0.5, c='red', s=1, marker='s', label=f'> {SPAD_MAX}')
    
    # 添加图例
    if len(x_low) > 0 or len(x_high) > 0:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(str(out_seaborn), dpi=600, transparent=False)
    plt.close()
    
    # 统计信息
    total_pixels = np.sum(mask)
    in_range_pixels = np.sum(in_range)
    low_pixels = np.sum(out_of_range_low)
    high_pixels = np.sum(out_of_range_high)
    
    print(f"[OK] {sample_name}: 伪彩色图已生成")
    print(f"  总像素: {total_pixels}, 范围内: {in_range_pixels}, <{SPAD_MIN}: {low_pixels}, >{SPAD_MAX}: {high_pixels}")
    
    return True

if __name__ == '__main__':
    print("=" * 80)
    print("生成SPAD伪彩色图（批量处理8个样本）")
    print(f"颜色条范围: {SPAD_MIN} - {SPAD_MAX}")
    print("=" * 80)
    print()
    
    success_count = 0
    for sample in SAMPLES:
        if generate_spad_map(sample):
            success_count += 1
    
    print()
    print(f"生成完成: {success_count}/{len(SAMPLES)} 个样本")
    print(f"图片保存在: {FIGURE_DIR}")
