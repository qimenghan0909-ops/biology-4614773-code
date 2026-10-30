import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold


def CARS_Cloud(X, y, N=20, f=30, cv=5):
    '''
    X : 光谱数据 (样本数 x 波长数)
    y :特征值
    N : 迭代次数（默认为20次）
    f : 主成分数量（默认为30）
    cv: 交叉验证数量（默认为5）

    return :
            OptWave: 筛选出的关键波长索引
    '''
    # 设置训练集比例（80%样本用于训练）
    p = 0.8
    # 获取数据形状：m=样本数，n=波长数
    m, n = X.shape
    # 计算波长数量衰减参数
    u = np.power((n / 2), (1 / (N - 1)))
    # 计算指数衰减率（控制波长减少速度）
    k = (1 / (N - 1)) * np.log(n / 2)
    # 计算训练样本数量（80%样本）
    cal_num = int(np.round(m * p))

    # 初始化所有波长的索引（0到n-1）
    wave_index = np.arange(n)
    # 用于记录每次迭代选择的波长
    WaveData = []
    # 用于记录每次迭代的误差
    RMSECV = []

    # 开始迭代循环（共N次）
    for i in range(1, N + 1):
        # 1. 随机选择训练样本
        cal_index = np.random.choice(m, size=cal_num, replace=False)

        # 2. 提取当前波长子集对应的训练数据
        xcal = X[cal_index][:, wave_index]
        ycal = y[cal_index]

        # 3. 交叉验证评估模型性能
        kf = KFold(n_splits=cv)  # 创建交叉验证分割器
        cv_errors = []  # 存储每次交叉验证的误差

        # 对每一折进行交叉验证
        for train_idx, test_idx in kf.split(xcal):
            # 创建PLS模型（主成分数不超过可用波长数）
            pls = PLSRegression(n_components=min(f, xcal.shape[1]))
            # 用训练数据拟合模型
            pls.fit(xcal[train_idx], ycal[train_idx])
            # 用模型预测测试数据
            y_pred = pls.predict(xcal[test_idx])
            # 计算并保存预测误差（均方误差）
            cv_errors.append(np.mean((ycal[test_idx] - y_pred.flatten()) ** 2))

        # 计算交叉验证的平均均方根误差（RMSE）
        RMSECV.append(np.sqrt(np.mean(cv_errors)))

        # 4. 记录当前选择的波长
        d = np.zeros(n, dtype=int)  # 创建全0数组（长度=总波长数）
        d[wave_index] = 1  # 将选择的波长位置设为1
        WaveData.append(d)  # 保存当前波长选择状态

        # 5. 如果不是最后一次迭代，更新波长选择
        if i < N:
            # 计算下一轮需要的波长数量（指数衰减公式）
            next_wave_num = max(1, int(round(u * np.exp(-k * i) * n)))

            # 6. 用全部训练数据计算回归系数
            pls_full = PLSRegression(n_components=min(f, xcal.shape[1]))  # 修复此处：删除了多余的括号
            pls_full.fit(xcal, ycal)
            # 获取回归系数（取绝对值表示重要性）
            coef = np.abs(pls_full.coef_.flatten())

            # 7. 根据系数重要性进行波长选择（轮盘赌算法）
            prob = coef / np.sum(coef)  # 计算选择概率
            # 根据概率选择下一轮波长
            selected = np.random.choice(
                len(wave_index),  # 从当前波长中选择
                size=next_wave_num,  # 选择的数量
                replace=False,  # 不重复选择
                p=prob  # 使用计算出的概率
            )
            # 更新波长索引
            wave_index = wave_index[selected]

    # 迭代结束后，找到误差最小的那次迭代
    MinIndex = np.argmin(RMSECV)
    # 获取最佳迭代对应的波长索引
    OptWave = np.where(WaveData[MinIndex] == 1)[0]

    # 返回筛选出的关键波长
    return OptWave