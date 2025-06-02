import os

import joblib
import pandas as pd
from sklearn import set_config
from sklearn.model_selection import StratifiedKFold, cross_val_predict

set_config(transform_output="pandas")

def run_pipeline(pipeline, X_test):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    final_proba = pd.DataFrame(y_proba, index=X_test.index)
    return final_proba

def save_model(models, cutoffs, location, name):
    results = {
        "models": models,
        "cutoffs": cutoffs
    }
    joblib.dump(results, os.path.join(location, f"model.{name}.pkl"))
    return None

def load_model(location, name):
    saved_model = joblib.load(os.path.join(location, f"model.{name}.pkl"))
    return saved_model['models'], saved_model['cutoffs']

def train_pipeline(pipeline, X, y: pd.Series, cv_splits=5, model_params=None):
    if hasattr(pipeline, 'build'):
        pipe = pipeline.build()  # 只有 pipeline builder 才调用 build()
    else:
        pipe = pipeline

    pipe, oof_proba, fold_results = run_cv_pipeline(pipe, X, y, cv_splits, model_params)
    return pipe, oof_proba, fold_results


def run_cv_pipeline(pipeline, X: pd.DataFrame, y: pd.Series, cv_splits=5, model_params=None):
    if model_params is None:
        model_params = {}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=model_params['smote_random_state'])
    pipe = pipeline
    # 交叉验证预测
    oof_proba = cross_val_predict(
        pipe, X, y, cv=cv, method='predict_proba', n_jobs=5
    )[:, 1]
    oof_predictions = pd.DataFrame(oof_proba, index=X.index)
    fold_predictions = {}
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        fold_predictions[f'fold_{fold + 1}'] = oof_predictions.iloc[val_idx]

    pipe.fit(X, y)
    pipe.fitted_features_ = X.columns.tolist()

    return pipe, oof_predictions, fold_predictions
