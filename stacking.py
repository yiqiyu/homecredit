from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#try:
#    from sklearn.impute import SimpleImputer as Imputer
#except ImportError:
#    from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Imputer
from sklearn.tree._tree import DTYPE

from sklearn.utils.validation import check_array
from sklearn.model_selection import KFold
import numpy as np
import gc
import copy
import pandas as pd

from feat_loading import load_dataframe, load_extra_feats_post, del_single_variance_and_nan_too_much


class StackingWrapper(object):
    def __init__(self, inited_clf, **fit_settings):
        self.clf = inited_clf
        self.fit_settings = fit_settings

    def fit(self, tnX, tny):
        return self.clf.fit(tnX, tny, **self.fit_settings)

    def __getattr__(self, item):
        return getattr(self.clf, item)


def get_oof(clf_proto, train, test, target_name, n_folds=5):
    train_ids = train['SK_ID_CURR']
    test_ids = test['SK_ID_CURR']
    test = test.drop(['SK_ID_CURR'], axis=1)
    y = train["TARGET"]
    X = train.drop(["TARGET", 'SK_ID_CURR'], axis=1)

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=250)
    test_predictions = np.zeros(test.shape[0])
    train_oof = np.zeros(X.shape[0])

    for i, (train_indices, predict_indices) in enumerate(k_fold.split(X)):
        print("%s/%s fold processing" % (i, n_folds))
        tnX, tny = X.loc[train_indices, :], y[train_indices]
        pX = X.loc[predict_indices, :]
        clf = copy.deepcopy(clf_proto)
        clf.fit(tnX, tny)
        train_oof[predict_indices] = clf.predict_proba(pX)[:, 1]
        test_predictions += clf.predict_proba(test)[:, 1] / k_fold.n_splits

        gc.enable()
        del tnX, tny, clf
        gc.collect()
        print("%s/%s fold finished" % (i, n_folds))

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
    # app_train = Imputer().fit_transform(app_train)
    # app_test = Imputer().fit_transform(app_test)
    # for dsname, ds in {"app_train": app_train, "app_test":app_test}.items():
    #     for col in ds:
    #         if ds[np.isinf(ds[col])].any():
    #             ds[col] = ds[col].replace(np.inf, 0)
    to_del = del_single_variance_and_nan_too_much(app_train)
    app_test = app_test.drop(to_del, axis=1)
    app_train = app_train.fillna(999999)
    app_test = app_test.fillna(999999)
    app_train = app_train.replace([np.inf, -np.inf], [99999, -99999])
    app_test = app_test.replace([np.inf, -np.inf], [99999, -99999])

    print("start training")
    rf_settings = dict(
        n_estimators=800,
        max_depth=10,
        min_samples_leaf=20,
        min_samples_split=10,
        min_impurity_decrease=1e-6,
        n_jobs=5,
        random_state=50,
        verbose=1
    )
    rf = RandomForestClassifier(**rf_settings)
    rf_train_oof, rf_test_feat = get_oof(rf, app_train, app_test, "RF")
    print("recording")
    rf_train_oof.to_csv("RF_oof.csv")
    rf_test_feat.to_csv("RF_test.csv")

    rf_test_feat.columns = ["SK_ID_CURR","TARGET"]
    rf_test_feat.to_csv("rf_submission.csv", index=False)
