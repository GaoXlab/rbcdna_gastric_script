from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np


class MeanClassifier(BaseEstimator, ClassifierMixin):
    """
    一个元分类器，接收一组分类器(pipeline)，输出它们预测值的均值

    参数:
    ----------
    classifiers : list of ClassifierMixin
        用于预测的一组分类器(pipeline)

    属性:
    ----------
    classes_ : array
        分类器预测的类别标签
    """

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):
        """
        拟合所有子分类器

        参数:
        ----------
        X : array-like, shape (n_samples, n_features)
            训练数据
        y : array-like, shape (n_samples,)
            目标值

        返回:
        ----------
        self : object
            返回self
        """
        # 拟合所有子分类器
        for clf in self.classifiers:
            clf.fit(X, y)

        # 获取类别标签
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """
        使用所有子分类器预测并返回平均预测值

        参数:
        ----------
        X : array-like, shape (n_samples, n_features)
            输入数据

        返回:
        ----------
        y_pred : array, shape (n_samples,)
            预测的目标值
        """
        check_is_fitted(self, ['classes_'])

        # 收集所有预测
        predictions = np.array([clf.predict(X) for clf in self.classifiers])

        # 计算平均预测值
        y_pred = np.mean(predictions, axis=0)

        return y_pred

    def predict_proba(self, X):
        """
        使用所有子分类器预测并返回平均概率

        参数:
        ----------
        X : array-like, shape (n_samples, n_features)
            输入数据

        返回:
        ----------
        proba : array, shape (n_samples, n_classes)
            预测的概率
        """
        check_is_fitted(self, ['classes_'])

        # 收集所有预测概率
        probas = np.array([clf.predict_proba(X) for clf in self.classifiers])

        # 计算平均概率
        avg_proba = np.mean(probas, axis=0)

        return avg_proba