# train_save.py
# 最优模型: D2+Lars+KS+LGBM (R²=0.7160)
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

# 使用与优化脚本完全相同的导入
from DataLoad.DataLoad import SetSplit
from Preprocessing.Preprocessing import Preprocessing_fit, Preprocessing_transform
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet_fit, SpctrumFeatureSelcet_transform
from lightgbm import LGBMRegressor
from Evaluate.RgsEvaluate import ModelRgsevaluate

# 设置基础目录
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'output' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

save_dir = str(MODEL_DIR)

def run_single_experiment(data, label, random_seed=42):
    """运行单次实验（与优化脚本完全一致）"""
    # 1. KS划分
    X_train, X_test, y_train, y_test = SetSplit('ks', data, label, test_size=0.2, randomseed=random_seed)
    
    # 2. D2预处理
    preprocess_params = Preprocessing_fit('D2', X_train)
    X_train_proc = Preprocessing_transform('D2', X_train, preprocess_params)
    X_test_proc = Preprocessing_transform('D2', X_test, preprocess_params)
    
    # 3. Lars特征选择
    feature_params = SpctrumFeatureSelcet_fit('Lars', X_train_proc, y_train, feature_num=30)
    X_train_feat = SpctrumFeatureSelcet_transform('Lars', X_train_proc, feature_params)
    X_test_feat = SpctrumFeatureSelcet_transform('Lars', X_test_proc, feature_params)
    
    # 4. LGBM训练
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.08,
        max_depth=-1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train_feat, y_train)
    
    # 5. 评估
    y_pred = model.predict(X_test_feat)
    y_train_pred = model.predict(X_train_feat)
    test_rmse, test_r2, test_mae = ModelRgsevaluate(y_pred, y_test)
    train_rmse, train_r2, train_mae = ModelRgsevaluate(y_train_pred, y_train)
    
    return test_r2, test_rmse, test_mae, train_r2, train_rmse, train_mae, model, feature_params, X_train, y_train

def run_single_experiment_safe(data, label, random_seed=42):
    """运行单次实验（带异常处理）"""
    try:
        return run_single_experiment(data, label, random_seed)
    except Exception as e:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, None, None, None)

def train_and_save_model(data, label, save_dir=save_dir):
    """
    训练 KS划分 → D2预处理 → Lars特征选择 → LGBM 模型并保存
    使用与优化脚本完全相同的数据流和并行计算，确保R2≈0.7160
    最优参数: Lars_nums=30, LGBM_n_estimators=500, LGBM_learning_rate=0.08
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    print(f"开始训练，使用10次并行运行（与优化脚本一致）")
    
    # 使用并行计算运行10次（与优化脚本一致）
    run_results = Parallel(n_jobs=20, verbose=0)(
        delayed(run_single_experiment_safe)(data, label, 42)
        for run in range(10)
    )
    
    # 过滤掉失败的结果
    valid_results = [r for r in run_results if r[0] is not np.nan]
    
    if len(valid_results) == 0:
        raise Exception("所有运行都失败了")
    
    # 计算平均值
    results_array = np.array([r[:6] for r in valid_results])
    mean_results = np.nanmean(results_array, axis=0)
    std_results = np.nanstd(results_array, axis=0)
    
    test_r2, test_rmse, test_mae, train_r2, train_rmse, train_mae = mean_results
    
    # 使用第一次运行的模型和特征（用于保存）
    model = valid_results[0][6]
    feature_params = valid_results[0][7]
    X_train = valid_results[0][8]
    y_train = valid_results[0][9]

    # 保存
    joblib.dump(feature_params, f'{save_dir}/lars_waves.pkl')
    joblib.dump(model, f'{save_dir}/lgbm_model.pkl')
    joblib.dump(X_train, f'{save_dir}/X_train.pkl')
    joblib.dump(y_train, f'{save_dir}/y_train.pkl')

    print(f"模型已保存到 {save_dir}/")
    print(f"Test: R2={test_r2:.4f}±{std_results[0]:.4f}, RMSE={test_rmse:.4f}±{std_results[1]:.4f}, MAE={test_mae:.4f}±{std_results[2]:.4f}")
    print(f"Train: R2={train_r2:.4f}±{std_results[3]:.4f}, RMSE={train_rmse:.4f}±{std_results[4]:.4f}, MAE={train_mae:.4f}±{std_results[5]:.4f}")
    
    return test_r2, test_rmse, test_mae, None, None, None, None

if __name__ == '__main__':
    # 读取光谱和标签（与优化脚本完全一致）
    data_df = pd.read_excel(DATA_DIR / "all_Spec.xlsx",
                            sheet_name='Sheet1', header=None)
    data = data_df.iloc[1:, :].values
    data = np.array(data, dtype=float)

    label_df = pd.read_excel(DATA_DIR / "all_SPAD.xlsx")
    label = label_df.iloc[:, 1].values
    label = np.array(label, dtype=float)

    data_name = "D2_Lars_ks_LGBM"
    
    # 训练并保存
    R2, RMSE, MAE, _, _, _, _ = train_and_save_model(data, label)
    print(f"\n训练完成 -> R2={R2:.4f}, RMSE={RMSE:.4f}, MAE={MAE:.4f}")

    # 保存结果
    results_df = pd.DataFrame({
        'Test_R2': [R2],
        'Test_RMSE': [RMSE],
        'Test_MAE': [MAE]
    })
    results_df.to_excel(MODEL_DIR / f'{data_name}_results.xlsx', index=False)
