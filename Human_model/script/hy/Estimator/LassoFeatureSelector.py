import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    """
    LASSO 特征选择器，可作为 scikit-learn Pipeline 的一个步骤

    参数:
    - eps: float, 默认为 1e-4
        alpha_min / alpha_max 的比率
    - n_alphas: int, 默认为 100
        沿正则化路径的 alpha 数量
    - cv: int, 默认为 5
        交叉验证折数
    - scale: bool, 默认为 True
        是否在 LASSO 之前标准化特征
    - positive: bool, 默认为 False
        是否强制系数为正
    - max_iter: int, 默认为 10000
        最大迭代次数
    - tol: float, 默认为 1e-4
        优化容忍度
    - random_state: int, 默认为 None
        随机种子
    - selection_threshold: float, 默认为 1e-5
        系数绝对值小于此值的特征将被丢弃
    """

    def __init__(self, eps=1e-4, n_alphas=100, cv=5, scale=False,
                 positive=False, max_iter=10000, tol=1e-4,
                 random_state=None, selection_threshold=1e-5):
        self.eps = eps
        self.n_alphas = n_alphas
        self.cv = cv
        self.scale = scale
        self.positive = positive
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.selection_threshold = selection_threshold

    def fit(self, X, y):
        """
        拟合模型并确定要保留的特征

        参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标变量 (n_samples,)

        返回:
        self: 拟合后的 transformer
        """
        # 检查输入
        X, y = check_X_y(X, y, y_numeric=True, multi_output=False)

        # 标准化数据
        if self.scale:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X

        # 拟合 LASSO CV 模型
        self.lasso_ = LassoCV(
            eps=self.eps,
            n_alphas=self.n_alphas,
            cv=self.cv,
            positive=self.positive,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )
        self.lasso_.fit(X_scaled, y)
        # 存储选择的特征
        self.coef_ = self.lasso_.coef_
        self.selected_features_ = np.where(
            np.abs(self.coef_) > self.selection_threshold
        )[0]
        if len(self.selected_features_) == 0:
            # print("没有选择任何特征，请调整 selection_threshold 参数，暂时无效化此次selector。")
            self.selected_features_ = np.where(
                np.abs(self.coef_) > -1
            )[0]
        # 存储最佳 alpha
        self.alpha_ = self.lasso_.alpha_
        return self

    def transform(self, X):
        """
        使用拟合的模型选择特征

        参数:
        X: 特征矩阵 (n_samples, n_features)

        返回:
        X_transformed: 只包含选择特征的矩阵 (n_samples, n_selected_features)
        """
        if len(self.selected_features_) == 0:
            print(X)
            return X
        # 检查是否已拟合
        check_is_fitted(self, ['lasso_', 'selected_features_', 'coef_'])

        # 检查输入
        X = check_array(X)
        # 标准化数据 (如果设置了 scale)
        if hasattr(self, 'scaler_'):
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        # 返回选择的特征
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        """
        拟合模型并转换数据

        参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标变量 (n_samples,)

        返回:
        X_transformed: 只包含选择特征的矩阵 (n_samples, n_selected_features)
        :param **kwargs:
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        获取输出特征名称

        参数:
        input_features: 输入特征名称列表 (可选)

        返回:
        输出特征名称数组
        """
        check_is_fitted(self, ['selected_features_'])

        if input_features is None:
            return np.array([f'feature_{i}' for i in self.selected_features_])

        if len(input_features) != self.coef_.shape[0]:
            raise ValueError(
                "input_features 的长度与训练时的特征数不匹配"
            )

        return np.array(input_features)[self.selected_features_]

    def get_support(self, indices=False):
        """
        获取选择的特征掩码或索引

        参数:
        indices: bool, 默认为 False
            如果为 True，返回索引；否则返回布尔掩码

        返回:
        选择的特征掩码或索引
        """
        check_is_fitted(self, ['selected_features_'])

        if indices:
            return self.selected_features_
        else:
            mask = np.zeros(len(self.coef_), dtype=bool)
            mask[self.selected_features_] = True
            return mask

    def plot_coefficients(self):
        """绘制特征系数图"""
        check_is_fitted(self, ['coef_', 'selected_features_'])

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.coef_)), self.coef_)
        plt.axhline(0, color='black', linewidth=1)
        plt.title('LASSO Coefficients')
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')

        # 标记选择的特征
        for idx in self.selected_features_:
            plt.bar(idx, self.coef_[idx], color='red')

        plt.show()


# 使用示例
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge

    # 生成示例数据
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=5,
        noise=0.5,
        random_state=42
    )

    # 创建特征名称
    feature_names = [f'X{i}' for i in range(X.shape[1])]

    # 创建并拟合 Pipeline
    pipeline = Pipeline([
        ('lasso_select', LassoFeatureSelector(cv=5, random_state=42)),
        ('ridge', Ridge())
    ])

    pipeline.fit(X, y)

    # 获取选择的特征
    selector = pipeline.named_steps['lasso_select']
    print(f"Selected features: {selector.get_feature_names_out(feature_names)}")
    print(f"Coefficients: {selector.coef_[selector.selected_features_]}")
    print(f"Optimal alpha: {selector.alpha_}")

    # 绘制系数图
    selector.plot_coefficients()