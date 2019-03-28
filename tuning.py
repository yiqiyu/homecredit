import os

from skopt import BayesSearchCV
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from feat_loading import *


class BayesSearchCV(BayesSearchCV):
    def _run_search(self, x): raise BaseException('Use newer skopt')


if __name__ == '__main__':
    print("start loading")
    app_train = load_dataframe("train_all_cat.csv")
    # app_test = load_dataframe("test_all_cat.csv")

    print("start catboost")
    cat_feats = np.where(app_train.dtypes == np.object)[0]
    app_train.loc[cat_feats, :].fillna("NaN", inplace=True)
    # app_test.loc[cat_feats, :].fillna("NaN", inplace=True)
    app_train = app_train.fillna(999999)
    # app_test = app_test.fillna(999999)
    cat_feats = cat_feats - 2

    cb_params = {
        # 'iterations': (300, 500),
        # 'learning_rate': (1e-2, 0.5, 'log-uniform'),
        'depth': (5, 15),
        'l2_leaf_reg': (1e-2, 1e+3, 'log-uniform'),
        'bootstrap_type': ['Bernoulli'],
        'subsample': (0.6, 1),
        'scale_pos_weight': [0.2],
        'eval_metric': ['AUC'],
        'metric_period': [50],
        'od_type': ['Iter'],
        'od_wait': [45],
        'allow_writing_files': [False],
        # "early_stopping_rounds": 300,
        "verbose": [200],
        "thread_count": [5],
        "random_strength": (1, 100)

    }
    cb = CatBoostClassifier(iterations=300, learning_rate=0.5, boosting_type="Plain", rsm=0.1)
    opt = BayesSearchCV(
        cb,
        cb_params,
        n_iter=32,
        fit_params=dict(cat_features=cat_feats),
        # n_jobs=2
    )
    X = app_train.drop(['SK_ID_CURR', "TARGET"], axis=1)
    y = app_train["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)
    opt.fit(X_train, y_train)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test, y_test))
    print(opt.cv_results_ )

    with open("wwwww.txt", "w+") as f:
        f.write(opt.best_score_)
        f.write("\n")
        f.write(opt.score(X_test, y_test))
        f.write("\n")
        f.write(opt.cv_results_ )


    # # print("start xgboost")
    # xg_params = dict(
    #     n_estimators=(100, 400),
    #     max_depth=(5, 20),
    #     colsample_bytree=(0.6, 1),
    #     reg_alpha=(0.01, 10),
    #     reg_lamda=(0.01, 10),
    #     learning_rate=(1e-2, 0.5, 'log-uniform'),
    #     min_child_weight=(10, 40),
    # )
    # #
    # xg = XGBClassifier(n_jobs=4, silent=False,eval_metric="auc",objective="binary:logistic",
    #     seed=50)
    # opt = BayesSearchCV(
    #     xg,
    #     xg_params,
    #     n_iter=32,
    #     fit_params=dict(early_stopping_rounds=200),
    #     # n_jobs=2
    # )
    # X = app_train.drop(['SK_ID_CURR', "TARGET"], axis=1)
    # y = app_train["TARGET"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)
    # opt.fit(X_train, y_train)
    #
    # print("val. score: %s" % opt.best_score_)
    # print("test score: %s" % opt.score(X_test, y_test))
    # print(opt.cv_results_)

    # os.system("shutdown -s -t  60 ")