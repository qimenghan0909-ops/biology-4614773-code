
# 导入必要的库和模块
import numpy as np
from sklearn.cross_decomposition import PLSRegression  # 导入偏最小二乘回归算法
import hpelm  # 导入极限学习机库
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR  # 导入支持向量回归算法
from sklearn.linear_model import LinearRegression  # 导入线性回归算法
from Evaluate.RgsEvaluate import ModelRgsevaluate  # 导入自定义的模型评估函数


'''
PLS算法解释：
偏最小二乘回归(PLS)是一种处理高维数据的回归方法。当输入特征很多且可能存在多重共线性时特别有效。
主成分数量(num)：决定模型复杂度。数量越多模型越复杂，但过多可能导致过拟合。
训练(fit)：模型学习输入特征(X)和目标值(y)之间的关系。
预测(predict)：使用学习到的关系对新数据做出预测。
评估指标：衡量预测值与真实值之间的差异。
'''
# 最小二乘(PLSR)
def Pls(X_train, X_test, y_train, y_test, num=None):
    # 2. 自动选择主成分数（如果未指定）
    if num is None:
        mse = []
        max_components = min(30, X_train.shape[1], X_train.shape[0])
        for i in range(1, max_components + 1):
            model = PLSRegression(n_components=i)
            scores = cross_val_score(model, X_train, y_train,
                                     scoring='neg_mean_squared_error', cv=5)
            mse.append(-np.mean(scores))
        num = np.argmin(mse) + 1
        print(f"自动选择的主成分数: {num}")

    # 创建PLS模型对象，设置主成分数量为num
    model = PLSRegression(n_components=num)
    # 使用训练数据训练模型
    # X_train是输入特征，y_train是目标值
    model.fit(X_train, y_train)
    # 使用训练好的模型对测试数据进行预测
    # X_test是测试集的特征数据
    #得到通过X_test预测出的目标值：y_pred
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    # 测试集指标
    test_rmse, test_r2, test_mae = ModelRgsevaluate(y_pred, y_test)
    # 训练集指标
    train_rmse, train_r2, train_mae = ModelRgsevaluate(y_train_pred, y_train)

    return test_rmse, test_r2, test_mae, y_pred, y_train_pred, train_rmse, train_r2, train_mae

'''
SVR算法解释：
支持向量回归(SVR)是一种基于支持向量机的回归方法，擅长处理非线性关系和小样本数据。
C参数：平衡模型的复杂度和对训练数据的拟合程度。值越大，模型越复杂。
gamma参数：影响单个样本对模型的影响范围。值越小，影响范围越大。
kernel(核函数)：用于将数据映射到高维空间以找到非线性关系。
'''

# 支持向量机回归(SVR)
def Svregression(X_train, X_test, y_train, y_test, C=2, gamma=1e-07, kernel='linear'):

    model = SVR(C=C, gamma=gamma, kernel=kernel)
    model.fit(X_train, y_train)
    # predict the values
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    # 测试集指标
    test_rmse, test_r2, test_mae = ModelRgsevaluate(y_pred, y_test)
    # 训练集指标
    train_rmse, train_r2, train_mae = ModelRgsevaluate(y_train_pred, y_train)
    return test_rmse, test_r2, test_mae, y_pred, y_train_pred, train_rmse, train_r2, train_mae

'''
极限学习机(ELM)是一种单隐层前馈神经网络，特点是训练速度快。
神经元：模型的基本计算单元，这里使用20个。
激活函数'sigm'：sigmoid函数，将输入映射到0-1之间。
训练：不同于传统神经网络，ELM的隐层参数是随机初始化的，只需调整输出层权重。
'''

# 极限学习机(ELM)
def ELM(X_train, X_test, y_train, y_test):
    # 创建ELM模型对象
    # X_train.shape[1]是输入特征的维度
    # 1是输出维度(单输出回归问题)
    model = hpelm.ELM(X_train.shape[1], 1)
    # 添加20个使用sigmoid激活函数的神经元
    model.add_neurons(20, 'sigm')

    # 训练模型
    # 'r'表示回归任务
    model.train(X_train, y_train, 'r')
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # 测试集指标
    test_rmse, test_r2, test_mae = ModelRgsevaluate(y_pred, y_test)
    # 训练集指标
    train_rmse, train_r2, train_mae = ModelRgsevaluate(y_train_pred, y_train)

    return test_rmse, test_r2, test_mae, y_pred, y_train_pred, train_rmse, train_r2, train_mae

'''
MLR算法解释：
多元线性回归是最基本的回归方法，假设目标值与输入特征之间存在线性关系。
线性关系：模型形式为 y = w1*x1 + w2*x2 + ... + b
训练：找到最佳的权重(w)和偏置(b)，使预测误差最小。
训练集预测：用于检查模型是否过拟合（在训练数据上表现很好但在新数据上表现差）。
'''

# 多元线性回归(MLR)
def MLR(X_train, X_test, y_train, y_test):
    # 建模
    model = LinearRegression()
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)

    y_train_pred = model.predict(X_train)
    # 测试集指标
    test_rmse, test_r2, test_mae = ModelRgsevaluate(y_pred, y_test)
    # 训练集指标
    train_rmse, train_r2, train_mae = ModelRgsevaluate(y_train_pred, y_train)

    return test_rmse, test_r2, test_mae, y_pred, y_train_pred, train_rmse, train_r2, train_mae

