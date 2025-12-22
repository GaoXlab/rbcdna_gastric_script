from sklearn.ensemble import VotingClassifier, RandomForestClassifier, VotingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier, ElasticNet, LinearRegression, Ridge, Lasso, \
    SGDRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Dict, Any, List, Tuple
from catboost import CatBoostClassifier, CatBoostRegressor
from .Estimator import *


class PipelineBuilder:
    def __init__(self, config: Dict[str, Any] = None):
        self._parent_builder = None
        self.config = config
        self.steps = []
        self._has_classifier = False  # 标记是否已添加classifier
        self._has_regressor = False  # 标记是否已添加regressor
        self._sub_pipelines = []  # 用于存储子pipeline(用于ensemble模型)

    def reconfigure(self, config: Dict[str, Any]) -> 'PipelineBuilder':
        """重新配置PipelineBuilder"""
        self.config = config
        return self

    def add_standard_scaler(self) -> 'PipelineBuilder':
        self.steps.append((
            'StandardScaler', StandardScaler()
        ))
        return self

    def add_robust_scaler(self) -> 'PipelineBuilder':
        self.steps.append((
            'RobustScaler', RobustScaler()
        ))
        return self

    def add_pca_feature_combiner(self) -> 'PipelineBuilder':
        """添加特征组合步骤"""
        pca_params = self.config.get('pca_params', {})

        pca_param_names = {
            'n_pcas': 10,
            'top_n': 5,
            'n_skip': 0,
            'scaler_name': 'StandardScaler',
            'from_pca': 1,
            'svd_solver': 'auto',
            'slice_pca': -1,
        }

        final_params = {}
        for param, default in pca_param_names.items():
            final_params[param] = pca_params.get(
                param,
                self.config.get(param, default)
            )
        self.steps.append((
            'pca_feature_combiner',
            PCABasedFeatureCombiner(**final_params)
        ))
        return self

    def add_xgb_shap_selector(self) -> 'PipelineBuilder':
        """添加XGBoost SHAP特征选择步骤"""
        # 确定任务类型（分类或回归）
        task_type = 'classification' if self._has_classifier else 'regression'

        self.steps.append((
            'xgb_shap_select',
            XGB_SHAP_FeatureSelector(
                task=task_type,
                n_splits=5,
                n_repeats=10,
                importance_threshold=self.config.get('importance_threshold', 'median'),
                random_state=1234,
                xgb_params=self.config.get('xgb_params', {}),
                verbose=False
            )
        ))
        return self

    def add_lasso_selector(self) -> 'PipelineBuilder':
        """添加Lasso特征选择步骤"""
        self.steps.append((
            'lasso_select',
            LassoFeatureSelector(cv=5, random_state=1234)
        ))
        return self

    def add_classifier(self, classifier) -> 'PipelineBuilder':
        """添加分类器"""
        if self._has_classifier or self._has_regressor:
            raise ValueError("Estimator already added to pipeline. Only one estimator is allowed.")
        self.steps.append(('classifier', classifier))
        self._has_classifier = True
        return self

    def add_regressor(self, regressor) -> 'PipelineBuilder':
        """添加回归器"""
        if self._has_classifier or self._has_regressor:
            raise ValueError("Estimator already added to pipeline. Only one estimator is allowed.")
        self.steps.append(('regressor', regressor))
        self._has_regressor = True
        return self

    # ================== 集成模型相关方法 ==================
    def start_sub_pipeline(self, name: str) -> 'PipelineBuilder':
        """开始一个新的子pipeline"""
        sub_builder = PipelineBuilder(self.config)
        sub_builder._parent_builder = self  # 设置父builder
        self._sub_pipelines.append({
            'name': name,
            'builder': sub_builder
        })
        return sub_builder

    def end_sub_pipeline(self) -> 'PipelineBuilder':
        """结束当前子pipeline并返回父builder"""
        if not self._sub_pipelines:
            return self._parent_builder if self._parent_builder else self
        return self._parent_builder  # 总是返回父builder

    def add_voting_classifier(self, estimators: List[Tuple[str, Any]] = [], voting: str = 'soft') -> 'PipelineBuilder':
        """添加投票分类器"""
        if not self._sub_pipelines:
            raise ValueError("No sub-pipelines defined for voting classifier")

        # 构建所有子pipeline
        sub_pipes = []
        for sub in self._sub_pipelines:
            sub_pipe = sub['builder'].build()
            sub_pipes.append((sub['name'], sub_pipe))

        self.steps.append((
            'ensemble',
            VotingClassifier(
                estimators=sub_pipes + estimators,
                voting=voting,
                n_jobs=-1,
                flatten_transform=True,
            )
        ))
        self._has_classifier = True

        return self

    def add_voting_regressor(self, estimators: List[Tuple[str, Any]] = []) -> 'PipelineBuilder':
        """添加投票回归器"""
        if not self._sub_pipelines:
            raise ValueError("No sub-pipelines defined for voting regressor")

        # 构建所有子pipeline
        sub_pipes = []
        for sub in self._sub_pipelines:
            sub_pipe = sub['builder'].build()
            sub_pipes.append((sub['name'], sub_pipe))

        self.steps.append((
            'ensemble',
            VotingRegressor(
                estimators=sub_pipes + estimators,
                n_jobs=-1
            )
        ))
        self._has_regressor = True
        return self

    def add_mean_classifier(self) -> 'PipelineBuilder':
        """添加均值分类器"""
        if not self._sub_pipelines:
            raise ValueError("No sub-pipelines defined for mean classifier")

        # 构建所有子pipeline
        sub_pipes = []
        for sub in self._sub_pipelines:
            sub_pipe = sub['builder'].build()
            sub_pipes.append(sub_pipe)

        self.steps.append((
            'ensemble',
            MeanClassifier(
                sub_pipes
            )
        ))
        self._has_classifier = True
        return self

    def add_mean_regressor(self) -> 'PipelineBuilder':
        """添加均值回归器"""
        if not self._sub_pipelines:
            raise ValueError("No sub-pipelines defined for mean regressor")

        # 构建所有子pipeline
        sub_pipes = []
        for sub in self._sub_pipelines:
            sub_pipe = sub['builder'].build()
            sub_pipes.append(sub_pipe)

        self.steps.append((
            'ensemble',
            MeanRegressor(
                sub_pipes
            )
        ))
        self._has_regressor = True
        return self

    def build(self) -> Pipeline:
        return ImbPipeline(self.steps)

    def get_pipeline_id(self) -> str:
        """生成pipeline的唯一标识符"""
        if not self.steps:
            return "empty_pipeline"

        # 获取所有步骤名称
        step_names = "+".join([name for name, _ in self.steps])

        # 获取估计器类名
        estimator_name = None
        for _, transformer in self.steps:
            if hasattr(transformer, 'predict'):  # 判断是否是估计器
                estimator_name = transformer.__class__.__name__
                break

        if not estimator_name:
            estimator_name = "NoEstimator"

        # 组合所有部分
        return f"{self.config.get('n', '')}_{self.config.get('n_pcas', '')}_{self.config.get('n_feas', '')}_{step_names}_{estimator_name}"

    # ================== 分类器方法 ==================
    def add_xgb_classifier(self) -> 'PipelineBuilder':
        """添加XGBoost分类器"""
        xgb_params = self.config.get('xgb_params', {})
        return self.add_classifier(XGBClassifier(**xgb_params))

    def add_elastic_net_classifier(self) -> 'PipelineBuilder':
        """添加弹性网络分类器"""
        elastic_net_params = self.config.get('elastic_net_params', {})
        return self.add_classifier(ElasticNet(**elastic_net_params))

    def add_lr_classifier(self) -> 'PipelineBuilder':
        """添加逻辑回归分类器"""
        lr_params = self.config.get('lr_params', {})
        return self.add_classifier(LogisticRegression(**lr_params))

    def add_sgd_classifier(self) -> 'PipelineBuilder':
        """添加SGD分类器"""
        sgd_params = self.config.get('sgd_params', {})
        return self.add_classifier(SGDClassifier(**sgd_params))

    def add_catboost_classifier(self) -> 'PipelineBuilder':
        """添加CatBoost分类器"""
        catboost_params = self.config.get('catboost_params', {})
        return self.add_classifier(CatBoostClassifier(**catboost_params))

    def add_lda_classifier(self) -> 'PipelineBuilder':
        """添加LDA分类器"""
        lda_params = self.config.get('lda_params', {})
        return self.add_classifier(LinearDiscriminantAnalysis(**lda_params))

    def add_rf_classifier(self) -> 'PipelineBuilder':
        """添加随机森林分类器"""
        rf_params = self.config.get('rf_params', {})
        return self.add_classifier(RandomForestClassifier(**rf_params))

    # ================== 回归器方法 ==================
    def add_linear_regressor(self) -> 'PipelineBuilder':
        """添加线性回归器"""
        linear_params = self.config.get('linear_params', {})
        return self.add_regressor(LinearRegression(**linear_params))

    def add_ridge_regressor(self) -> 'PipelineBuilder':
        """添加岭回归器"""
        ridge_params = self.config.get('ridge_params', {})
        return self.add_regressor(Ridge(**ridge_params))

    def add_lasso_regressor(self) -> 'PipelineBuilder':
        """添加Lasso回归器"""
        lasso_params = self.config.get('lasso_params', {})
        return self.add_regressor(Lasso(**lasso_params))

    def add_elastic_net_regressor(self) -> 'PipelineBuilder':
        """添加弹性网络回归器"""
        elastic_net_params = self.config.get('elastic_net_params', {})
        return self.add_regressor(ElasticNet(**elastic_net_params))

    def add_sgd_regressor(self) -> 'PipelineBuilder':
        """添加SGD回归器"""
        sgd_params = self.config.get('sgd_params', {})
        return self.add_regressor(SGDRegressor(**sgd_params))

    def add_xgb_regressor(self) -> 'PipelineBuilder':
        """添加XGBoost回归器"""
        xgb_params = self.config.get('xgb_params', {})
        return self.add_regressor(XGBRegressor(**xgb_params))

    def add_catboost_regressor(self) -> 'PipelineBuilder':
        """添加CatBoost回归器"""
        catboost_params = self.config.get('catboost_params', {})
        return self.add_regressor(CatBoostRegressor(**catboost_params))

    def add_rf_regressor(self) -> 'PipelineBuilder':
        """添加随机森林回归器"""
        rf_params = self.config.get('rf_params', {})
        return self.add_regressor(RandomForestRegressor(**rf_params))

    def add_svr_regressor(self) -> 'PipelineBuilder':
        """添加支持向量回归器"""
        from sklearn.svm import SVR
        svr_params = self.config.get('svr_params', {})
        return self.add_regressor(SVR(**svr_params))

    def add_knn_regressor(self) -> 'PipelineBuilder':
        """添加K近邻回归器"""
        from sklearn.neighbors import KNeighborsRegressor
        knn_params = self.config.get('knn_params', {})
        return self.add_regressor(KNeighborsRegressor(**knn_params))