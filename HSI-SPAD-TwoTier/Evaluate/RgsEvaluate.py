"""
只用到了第一个函数,ModelRgsevaluate
"""

# 导入必要的库
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # 导入评估指标
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库

'''
1. 基本评估函数 (ModelRgsevaluate)
    功能：评估回归模型的性能
    参数：
        y_pred: 模型预测值
        y_true: 真实值
    返回：
        RMSE: 均方根误差
        R²: 决定系数
        MAE: 平均绝对误差
'''
# 计算均方根误差(RMSE)，决定系数(R²)，平均绝对误差(MAE)
def ModelRgsevaluate(y_pred, y_true):

    mse = mean_squared_error(y_true, y_pred)
    R2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return np.sqrt(mse), R2, mae

'''
2、带数据逆变换的评估函数 (ModelRgsevaluatePro)
功能：对标准化后的数据进行逆变换后再评估
    参数：
        y_pred: 模型预测值（标准化后的）
        y_true: 真实值（标准化后的）
        yscale: 数据标准化器对象
    返回：
        RMSE: 逆变换后的均方根误差
        R²: 逆变换后的决定系数
        MAE: 逆变换后的平均绝对误差
'''



def ModelRgsevaluatePro(y_pred, y_true, yscale):

    yscaler = yscale
    # 将真实值逆变换回原始尺度
    y_true = yscaler.inverse_transform(y_true)
    # 将预测值逆变换回原始尺度
    y_pred = yscaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    R2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return np.sqrt(mse), R2, mae

'''
3. 回归结果可视化辅助函数 (ModelRgsEvaluatePlot)
功能：为回归结果可视化准备数据
    参数：
        X: 真实值（通常是横坐标）
        y: 预测值（通常是纵坐标）
    返回：
        index: 筛选后的数据索引
        model: 线性回归方程字符串
        coef: 回归系数
        intercept: 截距
        y_predict: 线性回归的预测值

'''
# 计算
def ModelRgsEvaluatePlot(X, y):
    # 确保X是二维数组（线性回归需要）
    X = X.reshape(-1, 1) #行数自动计算，列数为1
    # 确保X是二维数组（线性回归需要）
    reg = LinearRegression().fit(X, y)
    # 获取回归系数（斜率），保留4位小数
    coef = round(reg.coef_[0][0], 4)
    # 获取截距，保留3位小数
    intercept = round(reg.intercept_[0], 3)
    # 创建回归方程字符串，如"y=0.85x+1.234"
    model = str('y=') + str(coef) + 'x+' + str(intercept)
    # 确保X是二维数组
    X = X.reshape(-1, 1)
    # 使用线性回归模型进行预测
    y_predict = reg.predict(X)
    # 计算预测值与实际值的绝对差异
    delta = abs(y - y_predict)
    # 将差异转换为Pandas Series对象（类似一维数组）
    data = pd.Series(delta.reshape(1, -1)[0])
    # 找出差异小于4的数据点索引
    index = data[data < 4].index
    return index, model, coef, intercept, y_predict


'''
均方误差(MSE)：
计算预测值与真实值差异的平方的平均值
公式：MSE = (1/n) * Σ(预测值ᵢ - 真实值ᵢ)²
值越小越好，但单位是平方

均方根误差(RMSE)：
对MSE取平方根
公式：RMSE = √(MSE)
值越小越好，单位与原始数据相同

决定系数(R²)：
衡量模型解释数据变异性的比例
值在0-1之间，越接近1越好
公式：R² = 1 - (Σ(预测值ᵢ - 真实值ᵢ)² / Σ(真实值ᵢ - 平均值)²)

平均绝对误差(MAE)：
计算预测值与真实值绝对差异的平均值
公式：MAE = (1/n) * Σ|预测值ᵢ - 真实值ᵢ|
值越小越好，单位与原始数据相同
'''