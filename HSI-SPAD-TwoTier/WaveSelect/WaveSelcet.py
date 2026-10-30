from WaveSelect.Lar import Lar
from WaveSelect.Spa import SPA
from WaveSelect.Uve import UVE
from WaveSelect.Cars import CARS_Cloud
from WaveSelect.Pca import Pca
from WaveSelect.GA import GA
from sklearn.model_selection import train_test_split
import numpy as np

# ref1: 参考示例并做了修改
# ref2: https://github.com/FuSiry/OpenSA


def SpctrumFeatureSelcet(method, X, y, feature_num=30, N=20, f=30, cv=5):
    """

       :param method: 波长筛选/降维的方法，包括：Cars, Lars, Uve, Spa, Pca
       :param X: 光谱数据, shape (n_samples, n_features)
       :param y: 光谱数据对应标签：格式：(n_samples，)
       :param feature_num: 预测特征数量，要进行超参数优化
       :param N: CARS迭代次数
       :param f: CARS主成分数量
       :param cv: CARS交叉验证折数
       :return: X_Feature： 波长筛选/降维后的数据, shape (n_samples, n_features)
                y：光谱数据对应的标签, (n_samples，)
    """
    if method == "None":
        X_Feature = X
    elif method == "Cars":
        Featuresecletidx = CARS_Cloud(X, y, N=N, f=f, cv=cv)
        X_Feature = X[:, Featuresecletidx]
    elif method == "Lars":
        Featuresecletidx = Lar(X, y, nums=feature_num)
        # bands = 930+Featuresecletidx*6.32
        # print(bands)
        X_Feature = X[:, Featuresecletidx]
    elif method == "Uve":
        Uve = UVE(X, y, feature_num)
        Uve.calcCriteria()
        Uve.evalCriteria(cv=5)
        Featuresecletidx = Uve.cutFeature(X)
        X_Feature = Featuresecletidx[0]
    elif method == "Spa":
        Xcal, Xval, ycal, yval = train_test_split(X, y, test_size=0.2)
        Featuresecletidx = SPA().spa(
            Xcal=Xcal, ycal=ycal, m_min=8, m_max=50, Xval=Xval, yval=yval, autoscaling=1)
        X_Feature = X[:, Featuresecletidx]
    elif method == "GA":
        Featuresecletidx = GA(X, y)
        X_Feature = X[:, Featuresecletidx]
    elif method == "Pca":
        X_Feature = Pca(X)
    else:
        print("no this method of SpctrumFeatureSelcet!")

    return X_Feature, y


def SpctrumFeatureSelcet_fit(method, X_train, y_train, feature_num=30, N=20, f=30, cv=5, **kwargs):
    """
    特征选择fit函数
    
    :param method: 特征选择方法
    :param X_train: 训练集光谱数据
    :param y_train: 训练集标签
    :param feature_num: 特征数量（用于Lars等）
    :param N: CARS迭代次数
    :param f: CARS主成分数量
    :param cv: 交叉验证折数
    :param kwargs: 其他参数（如UVE_ncomp, GA_gen等）
    """
    if method == "None":
        return None
    elif method == "Cars":
        return CARS_Cloud(X_train, y_train, N=N, f=f, cv=cv)
    elif method == "Lars":
        return Lar(X_train, y_train, nums=feature_num)
    elif method == "Uve":
        # 支持UVE_ncomp参数
        uve_ncomp = kwargs.get('UVE_ncomp', feature_num)
        Uve = UVE(X_train, y_train, ncomp=uve_ncomp)
        Uve.calcCriteria()
        Uve.evalCriteria(cv=5)
        cuti = np.argmax(Uve.featureR2)
        selFeature = Uve.featureIndex[:cuti+1]
        return selFeature
    elif method == "Spa":
        Xcal, Xval, ycal, yval = train_test_split(X_train, y_train, test_size=0.2)
        return SPA().spa(
            Xcal=Xcal, ycal=ycal, m_min=8, m_max=50, Xval=Xval, yval=yval, autoscaling=1)
    elif method == "GA":
        # 支持GA_gen参数
        ga_gen = kwargs.get('GA_gen', 10)
        return GA(X_train, y_train, number_of_generation=ga_gen)
    elif method == "Pca":
        from sklearn.decomposition import PCA
        n_components = min(feature_num, X_train.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        return pca
    else:
        print("no this method of SpctrumFeatureSelcet!")
        return None


def SpctrumFeatureSelcet_transform(method, X, params):
    if method == "None":
        return X
    elif method == "Pca":
        return params.transform(X)
    else:
        return X[:, params]