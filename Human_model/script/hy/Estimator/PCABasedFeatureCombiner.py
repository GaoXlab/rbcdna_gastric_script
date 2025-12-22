import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler


class PCABasedFeatureCombiner(BaseEstimator, TransformerMixin):
    """基于PCA计算特征重要性，并组合原始特征与主成分"""

    def __init__(self, n_pcas=10, top_n=5, n_skip=0, scaler_name='StandardScaler', from_pca = 1, svd_solver='auto', slice_pca=-1):
        self.n_pcas = n_pcas  # PCA主成分数量
        self.top_n = top_n  # 保留的Top N原始特征
        if scaler_name == 'StandardScaler':
            self.scaler_ = StandardScaler()
        elif scaler_name == 'RobustScaler':
            self.scaler_ = RobustScaler()
        else:
            raise ValueError("Invalid scaler type. Choose 'StandardScaler' or 'RobustScaler'.")
        self.pca_ = PCA(n_components=n_pcas, random_state=1234, svd_solver=svd_solver)
        self.selected_features_ = []  # 存储选中的原始特征列名
        self.feature_importance_ = None
        self.loadings_ = None  # PCA载荷矩阵
        self.fitted_features_ = None
        self.n_skip = n_skip
        self.from_pca = from_pca
        self.scaler_name = scaler_name
        self.svd_solver = svd_solver
        self.slice_pca = slice_pca


    def fit(self, X, y=None):
        # 确保输入是DataFrame以获取列名（用于后续merge）
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.fitted_features_ = X.columns.tolist()
        # 1. 标准化数据

        X_scaled = self.scaler_.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        # 2. 训练PCA并计算特征重要性
        self.pca_.fit(X_scaled)
        self.loadings_ = self.pca_.components_  # 主成分对原始特征的权重
        r, c = self.pca_.components_.shape

        # 计算每个原始特征的综合重要性（绝对值求和）
        feature_importance = np.abs(self.loadings_)  # .sum(axis=0)
        feature_importance = pd.DataFrame(feature_importance, columns=X.columns,
                                          index=[f'PC{i + 1}' for i in range(r)]).T
        self.feature_importance_ = feature_importance

        # 选择Top N特征的列名
        self.selected_features_ = (
            self.feature_importance_
            .sort_values(by=f'PC{self.from_pca}', ascending=False)
            .head(self.top_n)
            .index.tolist()
        )
        return self

    def transform(self, X):
        # 确保输入是DataFrame以正确索引列
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        assert set(X.columns) == set(self.fitted_features_), "特征不匹配！"
        X = X[self.fitted_features_]
        # 1. 标准化原始数据
        X_scaled = self.scaler_.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        # 2. PCA转换得到主成分
        X_pca = self.pca_.transform(X_scaled)
        # 创建PCA特征DataFrame
        X_pca_df = pd.DataFrame(X_pca, index=X.index)
        X_pca_df.columns = [f'PC{i + 1}' for i in range(0, X_pca.shape[1])]

        if self.slice_pca != -1:
            X_pca_df = X_pca_df.iloc[:, :self.slice_pca]
        # 处理 n_skip
        if self.n_skip >= self.n_pcas:
            X_pca_df = pd.DataFrame(index=X.index)  # 跳过所有PCA
        elif self.n_skip > 0:
            X_pca_df = X_pca_df.iloc[:, self.n_skip:]

        # 4. 选择原始数据中的重要特征并合并
        X_selected = X[self.selected_features_].copy()
        X_combined = pd.concat([X_selected, X_pca_df], axis=1)
        # X_combined = pd.concat([X_pca_df, X_selected], axis=1)
        X_combined.index = X.index
        return X_combined
