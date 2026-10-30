import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置基础目录
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'output' / 'models'
FIGURE_DIR = BASE_DIR / 'output' / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 设置工作目录和文件路径
work_dir = str(MODEL_DIR)
fig_dir = str(FIGURE_DIR)

from sklearn.metrics import r2_score
# 确保输出目录存在
os.makedirs(fig_dir, exist_ok=True)

# 读取数据
y_test_true = pd.read_excel(os.path.join(work_dir, "all_y_test_0717.xlsx"), header=0).iloc[:, 0]
y_test_pred = pd.read_excel(os.path.join(work_dir, "all_y_test_pre_0717.xlsx"), header=0).iloc[:, 0]
y_train_true = pd.read_excel(os.path.join(work_dir, "all_y_train_0717.xlsx"), header=0).iloc[:, 0]
y_train_pred = pd.read_excel(os.path.join(work_dir, "all_y_train_pre_0717.xlsx"), header=0).iloc[:, 0]
r2_test = r2_score(y_test_true , y_test_pred)
r2_train = r2_score(y_train_true , y_train_pred)
print(r2_test)
print(r2_train)
# 创建图形
plt.figure(figsize=(6, 6), dpi=300, facecolor='white')
plt.rcParams['font.family'] = 'Times New Roman'

# 绘制对角线 (1:1线)
min_val = min(min(y_test_true), min(y_test_pred), min(y_train_true), min(y_train_pred))
max_val = max(max(y_test_true), max(y_test_pred), max(y_train_true), max(y_train_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.7)

# 绘制散点图
plt.scatter(y_train_true, y_train_pred, c='#1f77b4', alpha=0.8, s=90,
            edgecolor='w', linewidth=0.5, label='Train')
plt.scatter(y_test_true, y_test_pred, c='#ff7f0e', alpha=0.8, s=90,
            edgecolor='w', linewidth=0.5, label='Test')

# 设置标签
# plt.xlabel('Measured Value', fontsize=18, fontweight='bold')
# plt.ylabel('Predicted Value', fontsize=18, fontweight='bold')

plt.xlabel('Measured Value', fontsize=18)
plt.ylabel('Predicted Value', fontsize=18)
# 设置刻度向内 - 关键修改
plt.tick_params(axis='both', which='both', direction='in',
                top=False, right=False,  # 显示顶部和右侧刻度线
                labelsize=16,
                labeltop=False, labelright=False)  # 不显示顶部和右侧刻度值

# 设置坐标轴范围
buffer = 0.05 * (max_val - min_val)
plt.xlim(min_val - buffer, max_val + buffer)
plt.ylim(min_val - buffer, max_val + buffer)

# 添加图例
plt.legend(loc='upper left', fontsize=16, handletextpad=0)

# 调整布局
plt.tight_layout()

# 保存高质量图片
output_path = os.path.join(fig_dir, 'diagonal_plot.png')
plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')

# 显示图形
plt.show()

print(f"图像已保存至: {output_path}")
