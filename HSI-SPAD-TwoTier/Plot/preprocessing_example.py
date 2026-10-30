import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置基础目录
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
FIGURE_DIR = BASE_DIR / 'output' / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 设置工作目录和文件路径
work_dir = str(DATA_DIR)
fig_dir = str(FIGURE_DIR)

plt.rcParams['font.family'] = 'Times New Roman'

# 设置文件路径
preprocessing_methods = ["None", "MA", "SG", "MSC", "D1", "D2", "DT", "WAVE"]
for method in preprocessing_methods:
    file_name = f"{method}_preprocessed_with_header.xlsx"
    file_path = os.path.join(work_dir, file_name)
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 读取数据
        df = pd.read_excel(file_path)

        # 提取波长作为x轴（第一行数据）
        wavelengths = df.columns[1:].astype(float)  # 跳过第一列（如果有样本ID）

        # 准备绘图
        plt.figure(figsize=(12, 8))

        # 循环绘制每个样本
        for i in range(1, len(df)):  # 从第1行开始（跳过表头）
            sample_data = df.iloc[i, 1:].values.astype(float)  # 提取反射率数据
            plt.plot(wavelengths, sample_data, linewidth=2, alpha=0.7)

        # plt.xlabel('Wavelength(nm)', fontsize=50, labelpad=10)
        # plt.ylabel('Reflectance', fontsize=50,labelpad=10)

        #tick是表示刻度线

        plt.tick_params(axis='both', which='both', direction='in',
                        top=False, right=False,left=False,bottom=False,  # 显示顶部和右侧刻度线
                        labelsize=40,width=1.5,
                        labeltop=False, labelright=False,labelbottom=False, labelleft=False)  # 不显示顶部和右侧刻度值

        #spine表示坐标的边框
        for spine in plt.gca().spines.values():
            spine.set_linewidth(2)  # 设置线宽为2

        # 设置合理的x轴范围
        min_wave = min(wavelengths)
        max_wave = max(wavelengths)
        plt.xlim(min_wave - 10, max_wave + 10)

        # 保存图像
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, f'{method}spectral_curves.png'), dpi=600, bbox_inches='tight')

        plt.show()
