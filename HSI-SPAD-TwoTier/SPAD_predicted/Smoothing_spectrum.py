# D2预处理像素光谱（批量处理12个样本）
import pandas as pd
import numpy as np
from pathlib import Path
from Preprocessing.Preprocessing import D2

# 设置基础目录
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'output' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 像素光谱数据目录
PIXEL_DATA_DIR = Path(r'C:\Users\13004\Desktop\SPAD\fig_shili\spectra_data')

# 实际存在的8个样本列表
SAMPLES = [
    'CK_HS-45', 'CK_HS-17', 'DS_HS-17', 'DS_HS-103',
    'DS_HS-99', 'DS_HS-45', 'DS_HS-58', 'DS_HS-116'
]

def process_sample(sample_name):
    """对单个样本进行D2预处理"""
    file_name = f"{sample_name}_maize_pixel_spectra"
    input_path = PIXEL_DATA_DIR / f"{file_name}.xlsx"
    
    if not input_path.exists():
        print(f"文件不存在: {input_path}")
        return False
    
    # 读取像素光谱数据
    data_df = pd.read_excel(input_path, header=None)
    
    # 提取坐标
    y_coords = data_df.iloc[1:, 0].values
    x_coords = data_df.iloc[1:, 1].values
    
    # 提取光谱数据
    spectra = data_df.iloc[1:, 2:].values.astype(float)
    
    # 对光谱数据进行D2预处理（二阶导数）
    # D2会使维度减少2（300 -> 298）
    spectra_d2 = D2(spectra)
    
    # 创建包含坐标和D2处理后光谱数据的结果DataFrame
    # 获取波段名称（从原始数据第一行）
    band_names = data_df.iloc[0, 2:].values
    # D2处理后波段数减少2，需要调整列名
    columns = ['y', 'x'] + [f'D2_{i}' for i in range(spectra_d2.shape[1])]
    
    # 组合数据
    results_data = np.column_stack((y_coords, x_coords, spectra_d2))
    
    # 创建DataFrame
    results = pd.DataFrame(results_data, columns=columns)
    
    # 保存结果
    output_path = MODEL_DIR / f"{file_name}_spec_D2.xlsx"
    results.to_excel(output_path, index=False)
    print(f"[OK] {sample_name}: D2预处理完成 -> {output_path.name}")
    return True

if __name__ == '__main__':
    print("=" * 80)
    print("D2预处理像素光谱（批量处理12个样本）")
    print("=" * 80)
    print()
    
    success_count = 0
    for sample in SAMPLES:
        if process_sample(sample):
            success_count += 1
    
    print()
    print(f"处理完成: {success_count}/{len(SAMPLES)} 个样本")
