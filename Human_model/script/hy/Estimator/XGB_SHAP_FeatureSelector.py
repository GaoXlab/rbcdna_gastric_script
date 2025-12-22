import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor
import shap
import warnings


class XGB_SHAP_FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection based on XGBoost and SHAP importance with repeated CV.

    Parameters:
    -----------
    task : str, default='classification'
        Either 'classification' or 'regression'

    n_splits : int, default=5
        Number of folds for cross-validation

    n_repeats : int, default=10
        Number of times to repeat CV with different random feature orders

    importance_threshold : float or str, default='median'
        Threshold for selecting important features. Can be:
        - float: absolute threshold
        - 'median': median importance
        - 'mean': mean importance
        - 'top_k': select top k features (if int between 0-1, treated as percentage)

    random_state : int, default=None
        Random seed for reproducibility

    xgb_params : dict, default=None
        Parameters to pass to XGBoost

    verbose : bool, default=False
        Whether to print progress information
    """

    def __init__(self, task='classification', n_splits=5, n_repeats=10,
                 importance_threshold='median', random_state=None,
                 xgb_params=None, verbose=False):
        self.task = task
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.importance_threshold = importance_threshold
        self.random_state = random_state
        self.xgb_params = xgb_params or {}
        self.verbose = verbose
        self.feature_importances_ = None
        self.selected_features_ = None
        self.is_dataframe_ = False

    def fit(self, X, y):
        """
        Compute feature importances using XGBoost and SHAP with repeated CV.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)
            Target values

        Returns:
        --------
        self : object
            Fitted transformer
        """
        self.is_dataframe_ = isinstance(X, pd.DataFrame)

        if self.is_dataframe_:
            self.feature_names_ = X.columns.tolist()
            X_values = X.values
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
            X_values = X

        # Initialize importance storage
        importances = np.zeros(X_values.shape[1])


        # Repeat CV multiple times with different random feature orders
        for _ in range(self.n_repeats):
            # Shuffle features to reduce order bias
            idx = np.random.permutation(X_values.shape[1])
            X_shuffled = X_values[:, idx]

            # K-fold CV
            kf = KFold(n_splits=self.n_splits, shuffle=True,
                       random_state=self.random_state)

            for train_idx, val_idx in kf.split(X_shuffled):
                X_train, X_val = X_shuffled[train_idx], X_shuffled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Initialize and fit XGBoost model
                if self.task == 'classification':
                    model = XGBClassifier(**self.xgb_params)
                else:
                    model = XGBRegressor(**self.xgb_params)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train,
                              eval_set=[(X_val, y_val)],
                              verbose=False)

                # Calculate SHAP values using TreeExplainer
                explainer = shap.TreeExplainer(model)

                # Handle DataFrame vs array input for SHAP
                if self.is_dataframe_:
                    X_val_df = pd.DataFrame(X_val, columns=[f'tmp_{i}' for i in range(X_val.shape[1])])
                    shap_values = explainer.shap_values(X_val_df)
                else:
                    shap_values = explainer.shap_values(X_val)

                # For multi-class, take mean absolute SHAP across all classes
                if isinstance(shap_values, list):
                    shap_importance = np.mean([np.mean(np.abs(v), axis=0) for v in shap_values], axis=0)
                else:
                    shap_importance = np.mean(np.abs(shap_values), axis=0)

                # Store importances in original feature order
                importances[idx] += shap_importance

        # Average importances
        self.feature_importances_ = importances / (self.n_repeats * self.n_splits)

        # Select important features
        self._select_features()

        return self

    def _select_features(self):
        """Select features based on importance threshold."""
        if isinstance(self.importance_threshold, str):
            if self.importance_threshold == 'median':
                threshold = np.median(self.feature_importances_)
            elif self.importance_threshold == 'mean':
                threshold = np.mean(self.feature_importances_)
            elif self.importance_threshold.endswith('%'):
                # Treat as top k percentage
                k = float(self.importance_threshold[:-1]) / 100
                n_features = int(len(self.feature_importances_) * k)
                threshold = np.sort(self.feature_importances_)[-n_features]
            else:
                raise ValueError(f"Unknown threshold string: {self.importance_threshold}")
        else:
            threshold = self.importance_threshold

        self.selected_features_ = self.feature_importances_ >= threshold
        self.selected_indices_ = np.where(self.selected_features_)[0]

        if hasattr(self, 'feature_names_'):
            self.selected_feature_names_ = [
                self.feature_names_[i] for i in self.selected_indices_
            ]

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns:
        --------
        X_reduced : array-like of shape (n_samples, n_selected_features)
            Input data with only selected features
        """
        if self.is_dataframe_:
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, self.selected_indices_]
            else:
                # If input was DataFrame during fit but now we get array
                return pd.DataFrame(X).iloc[:, self.selected_indices_]
        else:
            if isinstance(X, pd.DataFrame):
                # If input was array during fit but now we get DataFrame
                return X.values[:, self.selected_indices_]
            else:
                return X[:, self.selected_indices_]

    def get_feature_importance_df(self):
        """Return feature importances as a DataFrame."""
        return pd.DataFrame({
            'feature': self.feature_names_,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)

    def fit_transform(self, X, y):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)