from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold
import numpy as np
import gc
import copy

from feat_loading import load_dataframe, load_extra_feats_post

class StackingWrapper(object):
    def __init__(self, inited_clf, **fit_settings):
        self.clf = inited_clf
        self.fit_settings = fit_settings

    def fit(self, tnX, tny):
        return self.clf.fit(tnX, tny, **self.fit_settings)

    def __getattr__(self, item):
        return getattr(self.clf, item)


def get_oof(clf_proto, train, test, n_folds=5):
    train_ids = train['SK_ID_CURR']
    test_ids = test['SK_ID_CURR']
    test = test.drop(['SK_ID_CURR'], axis=1)
    y = train["TARGET"]
    X = train.drop(["TARGET", 'SK_ID_CURR'], axis=1)

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=250)
    test_predictions = np.zeros(test.shape[0])
    train_oof = np.zeros(X.shape[0])

    for train_indices, predict_indices in k_fold.split(X):
        tnX, tny = X.loc[train_indices, :], y[train_indices]
        pX = X.loc[predict_indices, :]
        clf = copy.deepcopy(clf_proto)
        clf.fit(tnX, tny)
        train_oof[predict_indices] = clf.predict_proba(pX)[:, 1]
        test_predictions += clf.predict_proba(test)[:, 1] / k_fold.n_splits

        gc.enable()
        del tnX, tny, clf
        gc.collect()

    del X, y, train_ids, test_ids
    gc.collect()
    return train_oof, test_predictions


if __name__ == '__main__':
    app_train = load_dataframe("train_all.csv")
    app_test = load_dataframe("test_all.csv")
    app_train, app_test = load_extra_feats_post(app_train, app_test)

    rf_settings = dict(
        n_estimators=10000,
        max_depth=10,
        min_samples_leaf=20,
        min_samples_split=10,
        min_impurity_decrease=1e-6,
        n_jobs=4,
        random_state=50
    )
    rf = StackingWrapper(RandomForestClassifier(**rf_settings))
    rf_t_oof, t_feat = rf.get_oof(app_train, app_test)