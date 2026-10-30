from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import numpy as np


def Lar(X, y, nums=43):
    '''
           最小角回归
           X : 预测变量矩阵
           y ：标签
           nums : 选择的特征点的数目，默认为10
           return ：选择变量集的索引
    '''
    print("特征向量数：", nums)
    # 第一次Lars模型拟合（未标准化）
    # =================================================
    # 创建最小角回归模型对象
    Lars = linear_model.Lars()
    # 使用原始数据拟合模型（无预处理）
    Lars.fit(X, y)
    # 获取所有特征的回归系数（表示特征重要性）
    corflist = np.abs(Lars.coef_)
    # 3. 特征选择（第一次）
    # =================================================
    # 将系数列表转为numpy数组
    corf = np.asarray(corflist)
    # 选择最重要的nums个特征：
    # 1) argsort()返回从小到大排序的索引
    # 2) [-1:-(nums+1):-1] 逆序取前nums个最大值的索引
    SpectrumList = corf.argsort()[-1:-(nums+1):-1]
    # 对选择的特征索引进行升序排序
    SpectrumList = np.sort(SpectrumList)

    # 第二次Lars模型拟合（带标准化）
    # =================================================
    # 创建处理流水线：先标准化 -> 再Lars回归
    # with_mean=False: 不减去均值（适用于稀疏数据）
    model = make_pipeline(StandardScaler(with_mean=False), linear_model.Lars())  # 这里建立一个pipeline，先进行标准化，然后再进行Lars
    # 使用原始数据拟合流水线：
    # 1) 先对X进行标准化处理（缩放至单位方差）
    # 2) 再使用标准化后的数据训练Lars模型
    model.fit(X, y)
    # 从流水线中提取Lars模型的系数
    # named_steps['lars'] 获取流水线中的Lars步骤
    corflist = np.abs(model.named_steps['lars'].coef_)  # 这里获取coef_的时候需要通过named_steps来获取具体的步骤
    # 特征选择（第二次）
    # =================================================
    # 将系数列表转为numpy数组
    corf = np.asarray(corflist)
    SpectrumList = corf.argsort()[-1:-(nums + 1):-1]
    SpectrumList = np.sort(SpectrumList)

    return SpectrumList


'''
第二次覆盖了第一次的结果
'''