from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold
import numpy as np
import gc


class SkWrapper(object):
    def __init__(self, inited_clf, **fit_settings):
        self.clf = inited_clf
        self.fit_settings = fit_settings

    def fit(self, tnX, tny):
        return self.clf.fit(tnX, tny, **self.fit_settings)

    def __getattr__(self, item):
        return getattr(self.clf, item)


def get_oof(clf, train, test, n_folds=5):
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
        clf.fit(tnX, tny)
        train_oof[predict_indices] = clf.predict_proba(pX)[:, 1]
        test_predictions += clf.predict_proba(test)[:, 1] / k_fold.n_splits

    return train_oof, test_predictions