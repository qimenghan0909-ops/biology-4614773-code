import numpy as np  # 数值计算库，处理数组数据
import pandas as pd  # 数据分析库
import pywt  # 小波变换库，用于信号降噪
from scipy import signal  # 科学计算信号处理库 SG
from sklearn.linear_model import LinearRegression  # 线性回归模型
from copy import deepcopy  # 深度复制函数


# ref1: 参考示例并做了修改
# ref2: https://github.com/FuSiry/OpenSA


# 移动平均平滑
def MA(data, WSZ=11):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param WSZ: int
       :return: data after MA :(n_samples, n_features)
    """
    # data.shape返回一个元组，表示数组的维度
    # 对于光谱数据，通常是二维数组：(样本数, 波长点数)
    # data.shape[0] 表示样本的数量
    # 例如：如果有100个苹果的光谱数据，每个光谱有500个波长点，那么：
    # data.shape = (100, 500)
    # data.shape[0] = 100（样本数）
    # data.shape[1] = 500（波长点数）

    for i in range(data.shape[0]):
        #np.convolve(a, b)：计算两个数组的卷积，中间部分的移动平均值，长度 = 原始长度 - WSZ + 1
        out0 = np.convolve(data[i], np.ones(WSZ, dtype=int), 'valid') / WSZ # WSZ是窗口宽度，是奇数
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(data[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(data[i, :-WSZ:-1])[::2] / r)[::-1]
        data[i] = np.concatenate((start, out0, stop))
    return data


# Savitzky-Golay平滑滤波
def SG(data, w=11, p=2):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param w: int
       :param p: int
       :return: data after SG :(n_samples, n_features)
    """
    print("平滑窗口大小：", w)
    # 使用scipy的savgol_filter函数进行平滑
    return signal.savgol_filter(data, w, p)


# 一阶导数
def D1(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after First derivative :(n_samples, n_features)
    """
    n, p = data.shape # 获取样本数(n)和特征数(p)
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di


# 二阶导数
def D2(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after second derivative :(n_samples, n_features)
    """
    data = deepcopy(data)  # 创建数据副本（避免修改原始数据）
    if isinstance(data, pd.DataFrame): #检查对象的类型
        data = data.values
    # diff第一步：计算一阶导数
    temp2 = (pd.DataFrame(data)).diff(axis=1)
    # axis = 0 通常指的是沿着行的方向进行操作，而axis = 1 则指的是沿着列的方向进行操作
    temp3 = np.delete(temp2.values, 0, axis=1) # 删除第一列（NaN值）
    # 第二步：计算二阶导数
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)# 再次删除第一列
    return spec_D2


# 趋势校正(DT)
def DT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after DT :(n_samples, n_features)
    """
    lenth = data.shape[1]
    x = np.asarray(range(lenth), dtype=np.float32)
    out = np.array(data)
    l = LinearRegression()
    for i in range(out.shape[0]):
        l.fit(x.reshape(-1, 1), out[i].reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] - (j * k + b)

    return out


# 多元散射校正
def MSC(data,mean_spectrum=None):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MSC :(n_samples, n_features)
    """
    n, p = data.shape
    if mean_spectrum is None:
        mean_spectrum = np.mean(data, axis=0)

    msc = np.ones((n, p))
    for i in range(n):
        y = data[i, :]
        l = LinearRegression()
        l.fit(mean_spectrum.reshape(-1, 1), y.reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        msc[i, :] = (y - b) / k
    return msc


# 小波变换
def wave(data):
    """
        Wavelet Transform
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after wave :(n_samples, n_features)
    """
    data = deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    def wave_(data):
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)
        threshold = 0.04
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, 'db8')
        return datarec

    tmp = None
    for i in range(data.shape[0]):
        if (i == 0):
            tmp = wave_(data[i])
        else:
            tmp = np.vstack((tmp, wave_(data[i])))

    return tmp


def Preprocessing(method, data, window=11):

    if method == "None":
        data = data
    elif method == 'MA':
        data = MA(data)
    elif method == 'SG':
        data = SG(data, w=window)
    elif method == 'MSC':
        data = MSC(data)
    elif method == 'D1':
        data = D1(data)
    elif method == 'D2':
        data = D2(data)
    elif method == 'DT':
        data = DT(data)
    elif method == 'WAVE':
        data = wave(data)
    else:
        print("no this method of preprocessing!")

    return data


def Preprocessing_fit(method, data_train, window=11):
    params = {'method': method, 'window': window}
    if method == 'MSC':
        params['mean_spectrum'] = np.mean(data_train, axis=0)
    return params


def Preprocessing_transform(method, data, params=None, window=11):
    if method == "None":
        return data.copy()
    elif method == 'MA':
        return MA(data.copy(), WSZ=window)
    elif method == 'SG':
        return SG(data, w=window, p=2)
    elif method == 'MSC':
        return MSC(data, mean_spectrum=params['mean_spectrum'])
    elif method == 'D1':
        return D1(data)
    elif method == 'D2':
        return D2(data)
    elif method == 'DT':
        return DT(data)
    elif method == 'WAVE':
        return wave(data)
    else:
        print("no this method of preprocessing!")
        return data


