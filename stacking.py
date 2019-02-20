import pickle

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
#try:
#    from sklearn.impute import SimpleImputer as Imputer
#except ImportError:
#    from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree._tree import DTYPE

from sklearn.utils.validation import check_array
from sklearn.model_selection import KFold
import numpy as np
import gc
import copy
import pandas as pd

from feat_loading import *


class StackingWrapper(object):
    def __init__(self, inited_clf, **fit_settings):
        self.clf = inited_clf
        self.fit_settings = fit_settings

    def fit(self, tnX, tny):
        return self.clf.fit(tnX, tny, **self.fit_settings)

    def __getattr__(self, item):
        return getattr(self.clf, item)


def get_oof(clf_proto, train, test, target_name, n_folds=5, save_state=False, load_previous=False, **fit_params):
    print("start oof training")
    train_ids = train['SK_ID_CURR']
    test_ids = test['SK_ID_CURR']
    test = test.drop(['SK_ID_CURR'], axis=1)
    y = train["TARGET"]
    X = train.drop(["TARGET", 'SK_ID_CURR'], axis=1)

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=250)
    if not load_previous:
        test_predictions = np.zeros(test.shape[0])
        train_oof = np.zeros(X.shape[0])
    else:
        test_predictions = np.fromfile("states/test.bin")           # dtype needed?
        train_oof = np.fromfile("states/train.bin")
        with open("prev_state.bin", "rb") as f:
            prev_fold = pickle.load(f)

    for i, (train_indices, predict_indices) in enumerate(k_fold.split(X)):
        print("%s/%s fold processing" % (i+1, n_folds))
        if load_previous:
            if prev_fold >= i:
                print("done last time")
                continue

        tnX, tny = X.loc[train_indices, :], y[train_indices]
        pX = X.loc[predict_indices, :]
        clf = copy.deepcopy(clf_proto)
        clf.fit(tnX, tny, **fit_params)
        train_oof[predict_indices] = clf.predict_proba(pX)[:, 1]
        test_predictions += clf.predict_proba(test)[:, 1] / k_fold.n_splits
        if save_state:
            test_predictions.tofile("states/test.bin")
            train_oof.tofile("states/train.bin")
            with open("prev_state.bin", "wb") as f:
                pickle.dump(i, f)

        gc.enable()
        del tnX, tny, clf, pX
        gc.collect()
        print("%s/%s fold finished" % (i+1, n_folds))

    train_res = pd.DataFrame({'SK_ID_CURR': train_ids, target_name: train_oof})
    test_res = pd.DataFrame({'SK_ID_CURR': test_ids, target_name: test_predictions})
    del X, y, train_ids, test_ids
    gc.collect()
    return train_res, test_res


if __name__ == '__main__':
    print("start loading")
    app_train = load_dataframe("train_all.csv")
    app_test = load_dataframe("test_all.csv")
    app_train, app_test = load_extra_feats_post(app_train, app_test)
    to_del = del_single_variance_and_nan_too_much(app_train)
    app_test = app_test.drop(to_del, axis=1)
    # app_train = app_train.fillna(999999)
    # app_test = app_test.fillna(999999)
    # app_train = app_train.replace([np.inf, -np.inf], [99999, -99999])
    # app_test = app_test.replace([np.inf, -np.inf], [99999, -99999])

    # -----------------------------cat load-----------------------------------
    # app_train, app_test = load_all_tables("application_train.csv", "application_test.csv")
    # app_train.to_csv("train_all_cat.csv", index=False)
    # app_test.to_csv("test_all_cat.csv", index=False)
    # app_train = load_dataframe("train_all_cat.csv")
    # app_test = load_dataframe("test_all_cat.csv")

    # -------------------------------------Extra Tree--------------------------------------------
    # print("start training random forest")
    # rf_settings = dict(
    #     n_estimators=800,
    #     max_depth=10,
    #     min_samples_leaf=20,
    #     min_samples_split=10,
    #     min_impurity_decrease=1e-6,
    #     n_jobs=5,
    #     random_state=50,
    #     verbose=1
    # )
    # rf = RandomForestClassifier(**rf_settings)
    # rf_train_oof, rf_test_feat = get_oof(rf, app_train, app_test, "RF")
    # print("recording")
    # rf_train_oof.to_csv("RF_oof.csv")
    # rf_test_feat.to_csv("RF_test.csv")
    #
    # rf_test_feat.columns = ["SK_ID_CURR","TARGET"]
    # rf_test_feat.to_csv("rf_submission.csv", index=False)

    # # -------------------------------------Extra Tree--------------------------------------------
    # print("start training extra tree")
    # et_settings = dict(
    #     n_estimators=1200,
    #     max_depth=10,
    #     min_samples_leaf=20,
    #     min_samples_split=10,
    #     min_impurity_decrease=1e-6,
    #     n_jobs=5,
    #     random_state=50,
    #     verbose=1
    # )
    # et = ExtraTreesClassifier(**et_settings)
    # et_train_oof, et_test_feat = get_oof(et, app_train, app_test, "ET")
    # print("recording")
    # et_train_oof.to_csv("ET_oof.csv")
    # et_test_feat.to_csv("ET_test.csv")
    #
    # et_test_feat.columns = ["SK_ID_CURR","TARGET"]
    # et_test_feat.to_csv("et_submission.csv", index=False)

    # -------------------------------------Catboost--------------------------------------------
    # print("start catboost")
    # cat_feats = np.where(app_train.dtypes == np.object)[0]
    # app_train.loc[cat_feats, :].fillna("NaN", inplace=True)
    # app_test.loc[cat_feats, :].fillna("NaN", inplace=True)
    # app_train = app_train.fillna(999999)
    # app_test = app_test.fillna(999999)
    # cat_feats = cat_feats-2
    # cb_params = {
    #     'iterations':200,
    #     'learning_rate':0.1,
    #     'depth':10,
    #     'l2_leaf_reg':40,
    #     'bootstrap_type':'Bernoulli',
    #     'subsample':0.8,
    #     'scale_pos_weight':5,
    #     'eval_metric':'AUC',
    #     'metric_period':50,
    #     'od_type':'Iter',
    #     'od_wait':45,
    #     'allow_writing_files':False,
    #     # "early_stopping_rounds": 300,
    #     "verbose": 200,
    #     "thread_count": 5,
    #
    #     }
    #
    # cb = CatBoostClassifier(**cb_params)
    # cb_train_oof, cb_test_feat = get_oof(cb, app_train, app_test, "CB", cat_features=cat_feats)
    # print("recording")
    # cb_train_oof.to_csv("CB_oof.csv")
    # cb_test_feat.to_csv("CB_test.csv")
    #
    # cb_test_feat.columns = ["SK_ID_CURR","TARGET"]
    # cb_test_feat.to_csv("cb_submission.csv", index=False)

    # -------------------------------------xgboost--------------------------------------------
    # print("start xgboost")
    # xg_params = dict(
    #     n_estimators=150,
    #     max_depth=10,
    #     colsample_bytree=0.7,
    #     reg_alpha=0.2,
    #     reg_lamda=0.4,
    #     n_jobs=1,
    #     learning_rate=0.1,
    #     silent=False,
    #     min_child_weight=20,
    #     eval_metric="auc",
    #     objective="binary:logistic",
    #     seed=50
    # )
    #
    # xg = XGBClassifier(**xg_params)
    # xg_train_oof, xg_test_feat = get_oof(xg, app_train, app_test, "XG", early_stopping_rounds=200)
    # print("recording")
    # xg_train_oof.to_csv("XG_oof.csv")
    # xg_test_feat.to_csv("XG_test.csv")
    #
    # xg_test_feat.columns = ["SK_ID_CURR", "TARGET"]
    # xg_test_feat.to_csv("xg_submission.csv", index=False)


    # -----------------------------------------MLP---------------------------------------------
    print("start MLP")
    gc.enable()
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # train_ids = app_train['SK_ID_CURR']
    # test_ids = app_test['SK_ID_CURR']
    # y = app_train["TARGET"]
    # X = app_train.drop(["TARGET", 'SK_ID_CURR'], axis=1)
    # test = app_test.drop(['SK_ID_CURR'], axis=1)
    # test = scaler.transform(test)
    # scaler.fit(X, y)
    for col in app_train.columns:
        if col in ['SK_ID_CURR', "TARGET"]:
            continue
        if app_train[col].dtype == "bool":
            app_train[col] = app_train[col].replace([True, False], [1, -1])
            app_test[col] = app_test[col].replace([True, False], [1, -1])
        else:
            app_train[col] = (app_train[col] - app_train[col].mean())/app_train[col].std()
            app_test[col] = (app_test[col] - app_test[col].mean())/app_test[col].std()
        gc.collect()
    print("finished normalization")
    app_train.fillna(0, inplace=True)
    app_test.fillna(0, inplace=True)
    gc.collect()
    print("finished fillna")
    for col in app_train.columns:
        if col in ['SK_ID_CURR', "TARGET"]:
            continue
        if (not np.isfinite(app_train[col]).all()) or (not np.isfinite(app_test[col]).all()):
            app_train[col] = app_train[col].replace([np.inf, -np.inf], 0)
            app_test[col] = app_test[col].replace([np.inf, -np.inf],
                                                    0)
            gc.collect()

    # app_train.replace([np.inf, -np.inf], [1, -1], inplace=True)       # 内存爆炸
    # app_test.replace([np.inf, -np.inf], [1, -1], inplace=True)
    print("finished inf replacement")
    gc.collect()
    #
    # skb = SelectKBest(k=250)
    # skb.fit(X, y)

    mlp_params = dict(
        hidden_layer_sizes=(300,),
        learning_rate_init=0.01,
        learning_rate="adaptive",
        max_iter=200,
        early_stopping=True,
        random_state=50,
        verbose=True
    )
    mlp = MLPClassifier(**mlp_params)
    mlp_train_oof, mlp_test_feat = get_oof(mlp, app_train, app_test, "MLP",n_folds=5)
    print("recording")
    mlp_train_oof.to_csv("MLP_oof.csv")
    mlp_test_feat.to_csv("MLP_test.csv")

    mlp_test_feat.columns = ["SK_ID_CURR", "TARGET"]
    mlp_test_feat.to_csv("mlp_submission.csv", index=False)

    # -----------------------------------------KNN---------------------------------------------
    # print("start KNN")
    # for col in app_train.columns:
    #     if col in ['SK_ID_CURR', "TARGET"]:
    #         continue
    #     if app_train[col].dtype == "bool":
    #         app_train[col] = app_train[col].replace([True, False], [1, -1])
    #         app_test[col] = app_test[col].replace([True, False], [1, -1])
    #     else:
    #         app_train[col] = (app_train[col] - app_train[col].mean()) / app_train[col].std()
    #         app_test[col] = (app_test[col] - app_test[col].mean()) / app_test[col].std()
    #
    # app_train = app_train.replace([np.inf, -np.inf, np.nan], [1, -1, 0])
    # app_test = app_test.replace([np.inf, -np.inf, np.nan], [1, -1, 0])
    #
    #
    # knn_params = dict(
    #     n_neighbors=10,
    #     leaf_size=50,
    #     n_jobs=3
    # )
    # knn = KNeighborsClassifier(**knn_params)
    # KNN_train_oof, KNN_test_feat = get_oof(knn, app_train, app_test, "KNN")
    # print("recording")
    # KNN_train_oof.to_csv("knn_oof.csv")
    # KNN_test_feat.to_csv("knn_test.csv")
    #
    # KNN_test_feat.columns = ["SK_ID_CURR", "TARGET"]
    # KNN_test_feat.to_csv("knn_submission.csv", index=False)
