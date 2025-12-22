import os

import joblib
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, RepeatedStratifiedKFold


def run_pipeline(pipeline, X_test):
    if hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipeline.predict(X_test)
    final_proba = pd.DataFrame(y_proba, index=X_test.index)
    return final_proba

def save_model(models, cutoffs, n_feature, location, name):
    results = {
        "models": models,
        "cutoffs": cutoffs,
        "n_feature": n_feature,
    }
    joblib.dump(results, os.path.join(location, f"model.{name}.pkl"))
    return None

def load_model(location, name):
    saved_model = joblib.load(os.path.join(location, f"model.{name}.pkl"))
    return saved_model['models'], saved_model['cutoffs'], saved_model['n_feature']

def train_pipeline(pipeline, X, y: pd.Series, cv_splits=10, model_params=None):

    if hasattr(pipeline, 'build'):
        pipe = pipeline.build()  # 只有 pipeline builder 才调用 build()
    else:
        pipe = pipeline
    return run_cv_pipeline(pipe, X, y, cv_splits=cv_splits, model_params=model_params)

def run_cv_pipeline(pipeline, X: pd.DataFrame, y: pd.Series, cv_splits=10, model_params=None):
    if model_params is None:
        model_params = {}
    oof_predictions = pd.DataFrame(index=X.index)
    oof_predictions['value'] = 0
    oof_predictions['lr'] = 0
    oof_predictions['cb'] = 0
    fold_result = {}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=model_params['random_state'])
    pipe = pipeline
    # 交叉验证预测
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_valid = X.iloc[valid_idx]
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_result = pipe.predict_proba(X_valid)[:, 1]
        separate_result = pipe.transform(X_valid)
        oof_result_df = pd.DataFrame(oof_result, index=X_valid.index, columns=[0])
        oof_result_df['lr'] = separate_result[:, 1]
        oof_result_df['cb'] = separate_result[:, 3]
        oof_predictions.loc[X_valid.index, 'value'] += oof_result_df[0]
        oof_predictions.loc[X_valid.index, 'lr'] += oof_result_df['lr']
        oof_predictions.loc[X_valid.index, 'cb'] += oof_result_df['cb']
        fold_result[fold_idx] = oof_result_df
    oof_predictions.rename(columns={'value': 0}, inplace=True)
    pipe.fit(X, y)
    pipe.fitted_features_ = X.columns.tolist()

    return pipe, oof_predictions, fold_result