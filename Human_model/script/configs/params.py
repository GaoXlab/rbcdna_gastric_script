# 模型参数
MODEL_PARAMS = {
    'n': 1000,
    'n_pcas': 90,
    'smote_random_state': 1234,
    'lr_params': {
        'class_weight': 'balanced',
        'C': 1,
        'penalty': 'l2',
        'solver': 'liblinear'
    },
    'rf_params': {
        'class_weight': 'balanced', 'random_state': 1234,
    },
}