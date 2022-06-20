import os
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from lightgbm.sklearn import LGBMRegressor
from functools import partial
from sklearn.model_selection import KFold

def optimizer(trial, X, y, K):
    
    import os
    
    param = {
        'objective': 'regression', # 회귀
        'metric': 'mae',
        "random_state": 42,
        'max_depth': trial.suggest_int('max_depth', 8, 20),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-8, 1e-2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_loguniform('subsample', 0.4, 1)
        
        #"device" : 'gpu'
    }

    model = LGBMRegressor(**param, n_jobs=os.cpu_count())
    
    # K-Fold Cross validation을 구현합니다.
    folds = KFold(n_splits=K)
    scores = []
    
    for train_idx, val_idx in folds.split(X, y):
        X_train = X.iloc[train_idx, :]
        y_train = y.iloc[train_idx]
        
        X_val = X.iloc[val_idx, :]
        y_val = y.iloc[val_idx]
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=25)
        preds = model.predict(X_val)
        

        score = mean_absolute_error(y_val, preds)

        scores.append(score)
    
    
    # K-Fold의 평균 loss값을 돌려줍니다.
    return np.mean(scores)

    K = 4 # Kfold 수
def get_optimized_parameter(X, y, K):
    opt_func = partial(optimizer, X=X, y=y, K=K)
    lgbm_study = optuna.create_study(study_name="LGBM", direction="minimize") # regression task에서 mae를 최th화!
    lgbm_study.optimize(opt_func, n_trials=30)
    trial = lgbm_study.best_trial

    return trial.params