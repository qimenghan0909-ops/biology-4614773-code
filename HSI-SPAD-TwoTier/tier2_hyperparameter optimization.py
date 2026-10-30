"""
Top 6 Pipeline 超参数优化
对每种划分方式的前2个pipeline进行全面的超参数优化
包括：预处理参数 + 特征选择参数 + 模型参数
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import time
from datetime import datetime
from itertools import product

# 导入自定义模块
from DataLoad.DataLoad import SetSplit
from Preprocessing.Preprocessing import Preprocessing_fit, Preprocessing_transform
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet_fit, SpctrumFeatureSelcet_transform
from Evaluate.RgsEvaluate import ModelRgsevaluate

# 导入模型库
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings('ignore')

# 设置路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'data' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 模型实现 ====================

def RF(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None):
    """Random Forest"""
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42, 
        n_jobs=1  # 修复：改为1，避免CPU争用
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    test_rmse, test_r2, test_mae = ModelRgsevaluate(y_pred, y_test)
    train_rmse, train_r2, train_mae = ModelRgsevaluate(y_train_pred, y_train)
    return test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae


def LGBM(X_train, X_test, y_train, y_test, n_estimators=100, learning_rate=0.1):
    """LightGBM"""
    model = LGBMRegressor(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=-1, 
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    test_rmse, test_r2, test_mae = ModelRgsevaluate(y_pred, y_test)
    train_rmse, train_r2, train_mae = ModelRgsevaluate(y_train_pred, y_train)
    return test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae


def PLS(X_train, X_test, y_train, y_test, n_components=10):
    """PLS回归"""
    model = PLSRegression(n_components=n_components)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    test_rmse, test_r2, test_mae = ModelRgsevaluate(y_pred, y_test)
    train_rmse, train_r2, train_mae = ModelRgsevaluate(y_train_pred, y_train)
    return test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae


# ==================== 核心数据流 ====================

def run_single_experiment(data, label, pipeline_config, param_combo, random_seed=42):
    """运行单次实验"""
    # 解析参数
    params = dict(zip(pipeline_config['param_names'], param_combo))
    
    # 获取参数值
    split_method = pipeline_config['split']
    pre_method = pipeline_config['pre']
    feat_method = pipeline_config['feat']
    model_name = pipeline_config['model']
    
    # 预处理参数
    windows = params.get('MA_window', 11)
    
    # 特征选择参数
    feature_num = params.get('Lars_nums', 30)
    UVE_ncomp = params.get('UVE_ncomp', 30)
    
    # 模型参数
    RF_n_estimators = params.get('RF_n_estimators', 100)
    RF_max_depth = params.get('RF_max_depth', None)
    LGBM_n_estimators = params.get('LGBM_n_estimators', 100)
    LGBM_learning_rate = params.get('LGBM_learning_rate', 0.1)
    PLS_n_components = params.get('PLS_n_components', 10)
    
    # 1. 先划分数据集
    X_train, X_test, y_train, y_test = SetSplit(
        split_method, data, label, test_size=0.2, randomseed=random_seed
    )
    
    # 2. 预处理
    preprocess_params = Preprocessing_fit(pre_method, X_train, windows)
    X_train_proc = Preprocessing_transform(pre_method, X_train, preprocess_params, windows)
    X_test_proc = Preprocessing_transform(pre_method, X_test, preprocess_params, windows)
    
    # 3. 特征选择
    extra_kwargs = {}
    if feat_method == 'Uve':
        extra_kwargs['UVE_ncomp'] = UVE_ncomp
    
    feature_params = SpctrumFeatureSelcet_fit(
        feat_method, X_train_proc, y_train, feature_num=feature_num, **extra_kwargs
    )
    X_train_feat = SpctrumFeatureSelcet_transform(feat_method, X_train_proc, feature_params)
    X_test_feat = SpctrumFeatureSelcet_transform(feat_method, X_test_proc, feature_params)
    
    # 4. 建模
    if model_name == 'RF':
        test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae = RF(
            X_train_feat, X_test_feat, y_train, y_test,
            n_estimators=RF_n_estimators, max_depth=RF_max_depth
        )
    elif model_name == 'LGBM':
        test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae = LGBM(
            X_train_feat, X_test_feat, y_train, y_test,
            n_estimators=LGBM_n_estimators, learning_rate=LGBM_learning_rate
        )
    elif model_name == 'PLS':
        test_rmse, test_r2, test_mae, train_rmse, train_r2, train_mae = PLS(
            X_train_feat, X_test_feat, y_train, y_test,
            n_components=PLS_n_components
        )
    
    return test_r2, test_rmse, test_mae, train_r2, train_rmse, train_mae


def run_single_experiment_safe(data, label, pipeline_config, param_combo, random_seed=42):
    """运行单次实验（带异常处理）"""
    try:
        return run_single_experiment(data, label, pipeline_config, param_combo, random_seed)
    except Exception as e:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def optimize_pipeline(data, label, pipeline_config, pipeline_id, num_runs=10):
    """优化单个pipeline"""
    pipeline_name = f"{pipeline_config['pre']}+{pipeline_config['feat']}+{pipeline_config['split']}+{pipeline_config['model']}"
    
    # 生成所有参数组合
    param_values = list(pipeline_config['param_ranges'].values())
    param_combos = list(product(*param_values))
    
    print(f"\n{'='*80}")
    print(f"Pipeline {pipeline_id}: {pipeline_name}")
    print(f"参数组合数: {len(param_combos)}")
    print(f"运行次数: {num_runs}")
    print(f"总实验数: {len(param_combos) * num_runs}")
    print(f"{'='*80}")
    
    # 记录开始时间
    start_time = time.time()
    
    results = []
    
    # 对每个参数组合运行num_runs次（并行计算）
    for combo_idx, combo in enumerate(tqdm(param_combos, desc=f"优化 Pipeline {pipeline_id}")):
        # 使用并行计算运行num_runs次（所有运行使用相同种子42）
        run_results = Parallel(n_jobs=20, verbose=0)(
            delayed(run_single_experiment_safe)(data, label, pipeline_config, combo, 42)
            for run in range(num_runs)
        )
        
        # 计算平均值
        run_results = np.array(run_results)
        mean_results = np.nanmean(run_results, axis=0)
        std_results = np.nanstd(run_results, axis=0)
        
        # 保存结果
        result_dict = {
            'pipeline_id': pipeline_id,
            'pipeline_name': pipeline_name,
            'split_method': pipeline_config['split']
        }
        
        # 添加参数值
        for name, value in zip(pipeline_config['param_names'], combo):
            result_dict[name] = value
        
        # 添加结果
        result_dict['Test_R2_mean'] = mean_results[0]
        result_dict['Test_R2_std'] = std_results[0]
        result_dict['Test_RMSE_mean'] = mean_results[1]
        result_dict['Test_RMSE_std'] = std_results[1]
        result_dict['Test_MAE_mean'] = mean_results[2]
        result_dict['Test_MAE_std'] = std_results[2]
        result_dict['Train_R2_mean'] = mean_results[3]
        result_dict['Train_R2_std'] = std_results[3]
        result_dict['Train_RMSE_mean'] = mean_results[4]
        result_dict['Train_RMSE_std'] = std_results[4]
        result_dict['Train_MAE_mean'] = mean_results[5]
        result_dict['Train_MAE_std'] = std_results[5]
        
        results.append(result_dict)
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存当前pipeline的结果
    safe_name = pipeline_name.replace('+', '_')
    output_path = OUTPUT_DIR / f'pipeline{pipeline_id}_{safe_name}_optimization.xlsx'
    df.to_excel(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
    
    # 找出最优参数组合
    best_idx = df['Test_R2_mean'].idxmax()
    best_result = df.loc[best_idx]
    
    print(f"\n最优参数组合:")
    for name in pipeline_config['param_names']:
        print(f"  {name}: {best_result[name]}")
    print(f"  Test_R2: {best_result['Test_R2_mean']:.4f}±{best_result['Test_R2_std']:.4f}")
    print(f"  Test_RMSE: {best_result['Test_RMSE_mean']:.4f}±{best_result['Test_RMSE_std']:.4f}")
    print(f"\n耗时: {elapsed_time/60:.2f} 分钟")
    
    return df, best_result, elapsed_time


# ==================== 主程序 ====================

if __name__ == '__main__':
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("="*80)
    print("Top 6 Pipeline 超参数优化")
    print("="*80)
    print(f"开始时间: {start_datetime}")
    print(f"并行线程数: 20")
    print(f"随机种子: 42")
    
    # 加载数据
    print("\n加载数据...")
    data_path = DATA_DIR / 'all_Spec.xlsx'
    data_df = pd.read_excel(data_path, sheet_name='Sheet1', header=None)
    data = data_df.iloc[1:, :].values
    data = np.array(data, dtype=float)
    
    spad_path = DATA_DIR / 'all_SPAD.xlsx'
    spad_df = pd.read_excel(spad_path)
    label = spad_df.iloc[:, 1].values
    label = np.array(label, dtype=float)
    
    print(f"数据形状: {data.shape}")
    print(f"标签形状: {label.shape}")
    
    # 定义6个pipeline配置
    pipelines = [
        # RANDOM Top 2
        {
            'id': 1,
            'split': 'random', 'pre': 'D1', 'feat': 'Uve', 'model': 'RF',
            'param_names': ['UVE_ncomp', 'RF_n_estimators', 'RF_max_depth'],
            'param_ranges': {
                'UVE_ncomp': [10, 15, 20, 25, 30, 35, 40, 45, 50],
                'RF_n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                'RF_max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            }
        },
        {
            'id': 2,
            'split': 'random', 'pre': 'WAVE', 'feat': 'Lars', 'model': 'PLS',
            'param_names': ['Lars_nums', 'PLS_n_components'],
            'param_ranges': {
                'Lars_nums': [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
                'PLS_n_components': list(range(1, 31))
            }
        },
        # KS Top 2
        {
            'id': 3,
            'split': 'ks', 'pre': 'D2', 'feat': 'Lars', 'model': 'LGBM',
            'param_names': ['Lars_nums', 'LGBM_n_estimators', 'LGBM_learning_rate'],
            'param_ranges': {
                'Lars_nums': [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
                'LGBM_n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                'LGBM_learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            }
        },
        {
            'id': 4,
            'split': 'ks', 'pre': 'D1', 'feat': 'Uve', 'model': 'PLS',
            'param_names': ['UVE_ncomp', 'PLS_n_components'],
            'param_ranges': {
                'UVE_ncomp': [10, 15, 20, 25, 30, 35, 40, 45, 50],
                'PLS_n_components': list(range(1, 31))
            }
        },
        # SPXY Top 2
        {
            'id': 5,
            'split': 'spxy', 'pre': 'MA', 'feat': 'Lars', 'model': 'LGBM',
            'param_names': ['MA_window', 'Lars_nums', 'LGBM_n_estimators', 'LGBM_learning_rate'],
            'param_ranges': {
                'MA_window': [5, 7, 9, 11, 13, 15, 17, 19],
                'Lars_nums': [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
                'LGBM_n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                'LGBM_learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            }
        },
        {
            'id': 6,
            'split': 'spxy', 'pre': 'D1', 'feat': 'Lars', 'model': 'RF',
            'param_names': ['Lars_nums', 'RF_n_estimators', 'RF_max_depth'],
            'param_ranges': {
                'Lars_nums': [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
                'RF_n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                'RF_max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            }
        }
    ]
    
    # 计算总参数组合数
    total_combos = 0
    for p in pipelines:
        combos = len(list(product(*p['param_ranges'].values())))
        total_combos += combos
        print(f"Pipeline {p['id']}: {p['pre']}+{p['feat']}+{p['split']}+{p['model']} - {combos} 个组合")
    
    print(f"\n总参数组合数: {total_combos}")
    print(f"总实验数（10次运行）: {total_combos * 10}")
    
    # 优化每个pipeline
    all_results = []
    best_results = []
    time_records = []
    
    for pipeline in pipelines:
        pipeline_id = pipeline['id']
        
        df, best, elapsed_time = optimize_pipeline(data, label, pipeline, pipeline_id, num_runs=10)
        
        all_results.append(df)
        
        # 记录最优结果
        best_dict = {
            'pipeline_id': pipeline_id,
            'pipeline_name': best['pipeline_name'],
            'split_method': pipeline['split']
        }
        for param_name in pipeline['param_names']:
            best_dict[param_name] = best[param_name]
        best_dict['Test_R2'] = best['Test_R2_mean']
        best_dict['Test_R2_std'] = best['Test_R2_std']
        best_dict['Test_RMSE'] = best['Test_RMSE_mean']
        best_dict['Test_RMSE_std'] = best['Test_RMSE_std']
        best_dict['Test_MAE'] = best['Test_MAE_mean']
        best_dict['Test_MAE_std'] = best['Test_MAE_std']
        best_dict['Train_R2'] = best['Train_R2_mean']
        best_dict['Train_R2_std'] = best['Train_R2_std']
        best_dict['time_minutes'] = elapsed_time / 60
        
        best_results.append(best_dict)
        
        # 记录时间
        time_records.append({
            'pipeline_id': pipeline_id,
            'pipeline_name': best['pipeline_name'],
            'split_method': pipeline['split'],
            'elapsed_minutes': elapsed_time / 60
        })
    
    # 保存最优结果
    best_df = pd.DataFrame(best_results)
    best_output_path = OUTPUT_DIR / 'top6_optimization_best_results.xlsx'
    best_df.to_excel(best_output_path, index=False)
    print(f"\n最优结果已保存到: {best_output_path}")
    
    # 保存时间记录
    time_df = pd.DataFrame(time_records)
    time_output_path = OUTPUT_DIR / 'top6_optimization_time_records.xlsx'
    time_df.to_excel(time_output_path, index=False)
    print(f"时间记录已保存到: {time_output_path}")
    
    # 合并所有结果
    print("\n合并所有pipeline的结果...")
    all_df = pd.concat(all_results, ignore_index=True)
    all_output_path = OUTPUT_DIR / 'top6_optimization_all_results.xlsx'
    all_df.to_excel(all_output_path, index=False)
    print(f"所有结果已保存到: {all_output_path}")
    
    # 记录结束时间
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"开始时间: {start_datetime}")
    print(f"结束时间: {end_datetime}")
    print(f"总耗时: {total_time/3600:.2f} 小时 ({total_time/60:.2f} 分钟)")
    print(f"{'='*80}")
    
    # 打印最优结果汇总
    print("\n最优结果汇总:")
    print("-"*80)
    for _, row in best_df.iterrows():
        print(f"Pipeline {row['pipeline_id']}: {row['pipeline_name']}")
        print(f"  Test_R2: {row['Test_R2']:.4f}±{row['Test_R2_std']:.4f}")
        print(f"  Test_RMSE: {row['Test_RMSE']:.4f}±{row['Test_RMSE_std']:.4f}")
        print(f"  耗时: {row['time_minutes']:.2f} 分钟")
        print()
