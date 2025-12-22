# 模型参数
MODEL_PARAMS = {
    'random_state': 1234,
    'rf_params': {
        'class_weight':'balanced','random_state': 1234,
    },
    'xgb_params': {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'n_estimators': 190,
        'max_depth': 3,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 1,
        'gamma': 0.7,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'random_state': 1234,
        'n_jobs': 12
    },
    'catboost_params': {
        'thread_count': 24,
        'random_state': 1234,'verbose': 0,'depth': 8,'grow_policy': 'SymmetricTree',
                        'iterations': 500,'l2_leaf_reg': 5,'learning_rate': 0.1
    },
    'pca_params': {
        'svd_solver': "full",
        'top_n': 0,
    },
    'lr_params': {
        'class_weight': 'balanced',
        'C': 1,
        'penalty': 'l2',
        'solver': 'liblinear',
        'random_state': 1234,
    },
    'sgd_params': {
        'loss': 'log_loss',
        'penalty': 'elasticnet',
        'alpha': 0.001,
        'l1_ratio': 0.5,
        'max_iter': 2000,
        'tol': 1e-4,
        'learning_rate': 'optimal',
        'early_stopping': True,
        'validation_fraction': 0.1,
        'random_state': 1234
    }
}