# predict_new.py
# 使用最优模型 D2+Lars+KS+LGBM 进行像素级SPAD预测
# 方案A：对像素光谱先做MA平滑，再做D2预处理
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# 导入预处理函数
from Preprocessing.Preprocessing import MA, D2

# 设置基础目录
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'output' / 'models'
PIXEL_DATA_DIR = Path(r'C:\Users\13004\Desktop\SPAD\fig_shili\spectra_data')

# 实际存在的8个样本列表
SAMPLES = [
    'CK_HS-45', 'CK_HS-17', 'DS_HS-17', 'DS_HS-103',
    'DS_HS-99', 'DS_HS-45', 'DS_HS-58', 'DS_HS-116'
]
# SAMPLES = [
#     'CK_HS-17',  'DS_HS-116'
# ]

def predict_spectral(new_spectra, model_dir=str(MODEL_DIR)):
    """
    对新光谱进行预测
    new_spectra : (n_samples, n_wavelength)  D2预处理后的光谱
    return      : (n_samples,) 的预测值
    """
    # 1. 加载Lars选择的波长索引和LGBM模型
    selected_waves = joblib.load(f'{model_dir}/lars_waves.pkl')
    lgbm_model = joblib.load(f'{model_dir}/lgbm_model.pkl')

    # 2. 应用Lars选择的波长
    new_feature = new_spectra[:, selected_waves]

    # 3. 使用LGBM模型进行预测
    y_pred = lgbm_model.predict(new_feature)

    return y_pred.flatten()

def process_sample(sample_name):
    """处理单个样本：先MA平滑，再D2预处理，然后预测"""
    # 读取原始像素光谱数据
    file_name = f"{sample_name}_maize_pixel_spectra"
    input_path = PIXEL_DATA_DIR / f"{file_name}.xlsx"
    
    if not input_path.exists():
        print(f"✗ {sample_name}: 原始光谱文件不存在")
        return False
    
    # 读取原始光谱数据
    data_df = pd.read_excel(input_path, header=None)
    
    # 提取坐标
    y_coords = data_df.iloc[1:, 0].values
    x_coords = data_df.iloc[1:, 1].values
    
    # 提取光谱数据
    spectra = data_df.iloc[1:, 2:].values.astype(float)
    
    # 步骤1：MA平滑（减少像素光谱噪声）
    print(f"  对 {sample_name} 进行MA平滑...")
    spectra_ma = MA(spectra.copy(), WSZ=11)
    
    # 步骤2：D2预处理（与训练数据一致）
    print(f"  对 {sample_name} 进行D2预处理...")
    spectra_d2 = D2(spectra_ma)
    
    # 步骤3：预测
    print(f"  对 {sample_name} 进行SPAD预测...")
    y_hat = predict_spectral(spectra_d2)

    # 保存选择的波长索引和对应的实际波长值
    selected_waves = joblib.load(f'{MODEL_DIR}/lars_waves.pkl')
    
    # 读取训练数据的波长信息（第一行）
    train_data_df = pd.read_excel(DATA_DIR / "all_Spec.xlsx",
                                   sheet_name='Sheet1', header=None)
    wavelengths = train_data_df.iloc[0, 2:].values.astype(float)
    
    # D2处理后波长索引会偏移，需要调整
    # D2会使光谱维度减少2，索引从原来的300变为298
    # Lars选择的索引是基于D2处理后的298维数据
    selected_wavelengths = wavelengths[selected_waves + 1]  # 调整索引偏移
    
    # 保存波长信息（只保存一次）
    if sample_name == SAMPLES[0]:
        waves_df = pd.DataFrame({
            '序号': range(1, len(selected_waves) + 1),
            '波长索引': selected_waves,
            '实际波长值': selected_wavelengths
        })
        waves_df.to_excel(MODEL_DIR / f"lars_selected_waves.xlsx", index=False)
        print(f"  Lars波长索引已保存: {len(selected_waves)} 个波长")

    # 保存预测结果
    results = pd.DataFrame({
        'y': y_coords,
        'x': x_coords,
        'Predicted_SPAD': y_hat
    })
    output_path = MODEL_DIR / f"{file_name}_predictions_MA_D2.xlsx"
    results.to_excel(output_path, index=False)
    print(f"✓ {sample_name}: 预测完成 -> {output_path.name}")
    return True

if __name__ == '__main__':
    print("=" * 80)
    print("使用 D2+Lars+KS+LGBM 模型进行像素级SPAD预测（方案A：先MA平滑再D2）")
    print("=" * 80)
    print()
    
    success_count = 0
    for sample in SAMPLES:
        if process_sample(sample):
            success_count += 1
    
    print()
    print(f"预测完成: {success_count}/{len(SAMPLES)} 个样本")
