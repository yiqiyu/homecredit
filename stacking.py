from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold
import numpy as np
import gc


def get_oof(clf, train, test, n_folds=5):
    train_ids = train['SK_ID_CURR']
    test_ids = test['SK_ID_CURR']
    test = test.drop(['SK_ID_CURR'], axis=1)
    y = train["TARGET"]
    X = train.drop(["TARGET", 'SK_ID_CURR'], axis=1)
    feature_names = list(X.columns)

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=250)
    test_predictions = np.zeros(test.shape[0])
    train_predictions = np.zeros(train.shape[0])
    out_of_fold = np.zeros(X.shape[0])

    for train_indices, predict_indices in k_fold.split(X):
        tnX, tny = X.loc[train_indices, :], y[train_indices]
        pX = X.loc[predict_indices, :]
        clf.fit(tnX, tny)
        train_predictions[predict_indices] = clf.predict_proba(pX)[:, 1]
        test_predictions += clf.predict_proba(test)[:, 1] / k_fold.n_splits

        out_of_fold[valid_indices] = lgbm.predict_proba(vX, num_iteration=best_iteration)[:, 1]
    return train_predictions, test_predictions