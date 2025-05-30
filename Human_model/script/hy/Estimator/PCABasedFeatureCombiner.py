import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCABasedFeatureCombiner(BaseEstimator, TransformerMixin):
    """基于PCA计算特征重要性，并组合原始特征与主成分"""

    def __init__(self, n_components=10):
        self.n_components = n_components  # PCA主成分数量
        self.scaler_ = StandardScaler()
        self.pca_ = PCA(n_components=n_components, random_state=1234, svd_solver='full')
        self.fitted_features_ = None

    def fit(self, X, y=None):
        # 确保输入是DataFrame以获取列名
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.fitted_features_ = X.columns.tolist()

        # 标准化数据并训练PCA
        X_scaled = self.scaler_.fit_transform(X)
        self.pca_.fit(X_scaled)
        return self

    def transform(self, X):
        # 确保输入是DataFrame以正确索引列
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        assert set(X.columns) == set(self.fitted_features_), "特征不匹配！"
        X = X[self.fitted_features_]

        # 标准化原始数据并进行PCA转换
        X_scaled = self.scaler_.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # 2. PCA转换得到主成分
        X_pca = self.pca_.transform(X_scaled)
        # 创建PCA特征DataFrame
        X_pca_df = X_pca
        X_pca_df.columns = [f'PC{i + 1}' for i in range(1, X_pca.shape[1]+1)]

        return X_pca_df