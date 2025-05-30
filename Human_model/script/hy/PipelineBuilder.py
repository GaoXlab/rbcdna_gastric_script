from typing import Dict, Any

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List, Tuple

from .Estimator import *


class PipelineBuilder:
    def __init__(self, config: Dict[str, Any] = None):
        self._parent_builder = None
        self.config = config
        self.steps = []
        self._has_classifier = False  # 标记是否已添加classifier
        self._sub_pipelines = []  # 用于存储子pipeline(用于ensemble模型)

    def add_pca_feature_combiner(self) -> 'PipelineBuilder':
        """添加特征组合步骤"""
        self.steps.append((
            'pca_feature_combiner',
            PCABasedFeatureCombiner(self.config['n_pcas'])
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
        if self._has_classifier:
            raise ValueError("Classifier already added to pipeline. Only one classifier is allowed.")

        self.steps.append(('classifier', classifier))
        self._has_classifier = True
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
                n_jobs=-1
            )
        ))
        self._has_classifier = True
        return self

    def build(self) -> Pipeline:
        return ImbPipeline(self.steps)

    def add_lr_classifier(self) -> 'PipelineBuilder':
        """添加逻辑回归分类器，如果lr_params不存在则使用空字典"""
        lr_params = self.config.get('lr_params', {})
        return self.add_classifier(LogisticRegression(**lr_params))
    def add_rf_classifier(self) -> 'PipelineBuilder':
        """添加随机森林分类器，如果rf_params不存在则使用空字典"""
        rf_params = self.config.get('rf_params', {})
        return self.add_classifier(RandomForestClassifier(**rf_params))