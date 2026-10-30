"""
Tier 1: 新模型测试脚本
使用PLS、SVR、LGBM、RF进行高光谱SPAD预测
数据流：划分→预处理→特征选择→建模
随机种子：42
并行数：20
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import time
from datetime import datetime

# 导入自定义模块
from DataLoad.DataLoad import SetSplit
from Preprocessing.Preprocessing import Preprocessing_fit, Preprocessing_transform
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet_fit, SpctrumFeatureSelcet_transform
from Evaluate.RgsEvaluate import ModelRgsevaluate

# 导入模型库
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# 设置路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'data' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 模型实现 ====================

def RF(X_train, X_test, y_train, y_test):
    """Random Forest"""
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    test_rmse, test_r2, test_mae = ModelRgsevaluate(y_pred, y_test)
    train_rmse, train_r2, train_mae = ModelRgsevaluate(y_train_pred, y_train)
    return test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae


def LGBM(X_train, X_test, y_train, y_test):
    """LightGBM"""
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    test_rmse, test_r2, test_mae = ModelRgsevaluate(y_pred, y_test)
    train_rmse, train_r2, train_mae = ModelRgsevaluate(y_train_pred, y_train)
    return test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae


def PLS(X_train, X_test, y_train, y_test):
    """PLS回归（自动选择主成分数）"""
    # 使用交叉验证自动选择主成分数
    mse = []
    max_components = min(30, X_train.shape[1], X_train.shape[0])
    for i in range(1, max_components + 1):
        model = PLSRegression(n_components=i)
        scores = cross_val_score(model, X_train, y_train,
                                 scoring='neg_mean_squared_error', cv=5)
        mse.append(-np.mean(scores))
    num = np.argmin(mse) + 1
    
    model = PLSRegression(n_components=num)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    test_rmse, test_r2, test_mae = ModelRgsevaluate(y_pred, y_test)
    train_rmse, train_r2, train_mae = ModelRgsevaluate(y_train_pred, y_train)
    return test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae


def SVR_model(X_train, X_test, y_train, y_test):
    """SVR回归"""
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    test_rmse, test_r2, test_mae = ModelRgsevaluate(y_pred, y_test)
    train_rmse, train_r2, train_mae = ModelRgsevaluate(y_train_pred, y_train)
    return test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae


# 模型映射
model_dict = {
    "PLS": PLS,
    "SVR": SVR_model,
    "LGBM": LGBM,
    "RF": RF
}


# ==================== 核心数据流 ====================

def SpectralQuantitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, 
                                  SetSplitMethods, model_name, windows, feature_num, random_seed):
    """
    数据流顺序：
    1. 先划分数据集（使用原始光谱）
    2. 预处理：从训练集学习参数
    3. 特征选择：从训练集学习波长索引
    4. 建模：使用新模型
    """
    # 1. 先划分数据集（使用原始光谱和SPAD）
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, data, label, 
                                                 test_size=0.2, randomseed=random_seed)
    
    # 2. 预处理：从训练集学习参数
    preprocess_params = Preprocessing_fit(ProcessMethods, X_train, windows)
    X_train_proc = Preprocessing_transform(ProcessMethods, X_train, preprocess_params, windows)
    X_test_proc = Preprocessing_transform(ProcessMethods, X_test, preprocess_params, windows)
    
    # 3. 特征选择：从训练集学习波长索引
    feature_params = SpctrumFeatureSelcet_fit(FslecetedMethods, X_train_proc, y_train, feature_num)
    X_train_feat = SpctrumFeatureSelcet_transform(FslecetedMethods, X_train_proc, feature_params)
    X_test_feat = SpctrumFeatureSelcet_transform(FslecetedMethods, X_test_proc, feature_params)
    
    # 4. 建模（使用新模型）
    model_func = model_dict[model_name]
    test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae = model_func(
        X_train_feat, X_test_feat, y_train, y_test)
    
    return test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae


def run_experiment(reg_model, pre_method, key_band_method, split_method, data, label, 
                   windows, feature_num, random_seed):
    """运行单个实验并返回结果"""
    test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae = SpectralQuantitativeAnalysis(
        data, label, pre_method, key_band_method, split_method, reg_model, 
        windows, feature_num, random_seed
    )
    return [test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae,
            pre_method, key_band_method, split_method, reg_model, random_seed]


def run_full_experiment(data, label, regression_model, preprocessing_methods,
                        split_data_methods, WaveSelect_methods, feature_num,
                        windows, file_name, data_name, run_id, seeds):
    """运行完整的实验流程（一次运行）"""
    print(f"\n{'=' * 50}")
    print(f"开始运行 #{run_id + 1}，随机种子: {seeds[run_id]}")
    print(f"{'=' * 50}")

    # 生成所有参数组合
    param_combinations = [
        (reg_model, pre_method, key_band_method, split_method)
        for reg_model in regression_model
        for pre_method in preprocessing_methods
        for split_method in split_data_methods
        for key_band_method in WaveSelect_methods
    ]

    # 使用并行计算运行所有实验（20线程）
    results = Parallel(n_jobs=20, verbose=10)(
        delayed(run_experiment)(
            reg_model, pre_method, key_band_method, split_method,
            data, label, windows, feature_num, seeds[run_id]
        )
        for reg_model, pre_method, key_band_method, split_method in tqdm(
            param_combinations,
            total=len(param_combinations),
            desc=f"运行 #{run_id + 1} 进度"
        )
    )

    # 创建结果DataFrame并添加列名
    results_df = pd.DataFrame(
        results,
        columns=['Test_RMSE', 'Test_R2', 'Test_MAE', 'Train_RMSE', 'Train_R2', 'Train_MAE',
                 'Preprocessing', 'FeatureSelection', 'SplitMethod', 'RegressionModel', 'RandomSeed']
    )

    # 添加运行ID列
    results_df['RunID'] = run_id + 1

    # 保存结果到Excel
    output_path = f'data/output/new_models_{file_name}_run{run_id + 1}.xlsx'
    results_df.to_excel(output_path, index=False)
    print(f"运行 #{run_id + 1} 完成! 结果已保存到: {output_path}")

    return results_df


# ==================== 主程序 ====================

if __name__ == '__main__':
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("=" * 80)
    print("Tier 1: 模型测试（PLS、SVR、LGBM、RF）")
    print("=" * 80)
    print(f"开始时间: {start_datetime}")
    print(f"并行线程数: 20")
    print(f"随机种子: 42")
    
    file_names = ["all"]
    data_name = "new_models"
    num_runs = 10
    seeds = [42] * 10  # 所有运行使用种子42

    # 存储所有运行的结果
    all_results = []

    for file_name in file_names:
        # 加载数据
        data_path = DATA_DIR / f"{file_name}_Spec.xlsx"
        data_df = pd.read_excel(data_path, sheet_name='Sheet1', header=None)
        data = data_df.iloc[1:, :].values
        data = np.array(data)

        # 加载SPAD信息
        df = pd.read_excel(DATA_DIR / f"{file_name}_SPAD.xlsx")
        column_data = df.iloc[:, 1].values
        label = np.array(column_data)

        # 实验设置
        regression_model = ["PLS", "SVR", "LGBM", "RF"]
        preprocessing_methods = ["None", "MA", "SG", "MSC", "D1", "D2", "DT", "WAVE"]
        split_data_methods = ["random", "spxy", "ks"]
        WaveSelect_methods = ["None", "Cars", "Lars", "Uve", "Spa", "GA", "Pca"]
        
        feature_num = 30
        windows = 11

        # 计算总组合数
        total_combinations = (len(regression_model) * len(preprocessing_methods) * 
                             len(split_data_methods) * len(WaveSelect_methods))
        print(f"\n总组合数: {total_combinations}")
        print(f"运行次数: {num_runs}")
        print(f"总实验数: {total_combinations * num_runs}")

        # 运行多次实验
        for run_id in range(num_runs):
            results_df = run_full_experiment(
                data, label, regression_model, preprocessing_methods,
                split_data_methods, WaveSelect_methods, feature_num,
                windows, file_name, data_name, run_id, seeds
            )
            all_results.append(results_df)

        # 合并所有运行的结果
        combined_results = pd.concat(all_results, ignore_index=True)

        # 保存合并后的结果
        combined_output_path = f'data/output/{file_name}_new_models_combined_{num_runs}runs.xlsx'
        combined_results.to_excel(combined_output_path, index=False)
        print(f"\n所有{num_runs}次运行完成! 合并结果已保存到: {combined_output_path}")

    # 记录结束时间
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = end_time - start_time
    
    print(f"\n{'=' * 80}")
    print(f"开始时间: {start_datetime}")
    print(f"结束时间: {end_datetime}")
    print(f"总耗时: {total_time/3600:.2f} 小时 ({total_time/60:.2f} 分钟)")
    print(f"{'=' * 80}")
