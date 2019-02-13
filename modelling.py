import gc

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def model(train, test, cat_indices="auto", n_folds=5, parallel=3, need_oof=False):
    train_ids = train['SK_ID_CURR']
    test_ids = test['SK_ID_CURR']
    test = test.drop(['SK_ID_CURR'], axis=1)
    y = train["TARGET"]
    X = train.drop(["TARGET", 'SK_ID_CURR'], axis=1)
    feature_names = list(X.columns)

    k_fold = KFold(n_splits=5, shuffle=True, random_state=50)
    feature_importance_values = np.zeros(len(feature_names))
    test_predictions = np.zeros(test.shape[0])
    out_of_fold = np.zeros(X.shape[0])

    valid_scores = []
    train_scores = []
    best_iteration = 2200
    for train_indices, valid_indices in k_fold.split(X):
        # Training data for the fold
        tnX, tny = X.loc[train_indices, :], y[train_indices]
        # Validation data for the fold
        vX, vy = X.loc[valid_indices, :], y[valid_indices]

        lgbm = LGBMClassifier(boosting_type="goss", n_estimators=10000, objective='binary',
                              learning_rate=0.02, n_jobs=parallel, random_state=50,
                              subsample=0.81, reg_alpha=0.1, reg_lambda=0.1, min_child_samples=30,
                              max_depth=8, colsample_bytree=0.92, num_leaves=35
                              )

        lgbm.fit(tnX, tny, eval_metric="auc", categorical_feature=cat_indices, feature_name='auto',
                 eval_set=[(vX, vy), (tnX, tny)], eval_names=['valid', 'train'],
                 early_stopping_rounds=300, verbose=200)
        test_predictions += lgbm.predict_proba(test, num_iteration=best_iteration)[:, 1] / k_fold.n_splits

        best_iteration = lgbm.best_iteration_
        feature_importance_values += lgbm.feature_importances_ / k_fold.n_splits
        out_of_fold[valid_indices] = lgbm.predict_proba(vX, num_iteration=best_iteration)[:, 1]

        valid_scores.append(lgbm.best_score_['valid']['auc'])
        train_scores.append(lgbm.best_score_['train']['auc'])

        gc.enable()
        del lgbm, tnX, vX
        gc.collect()

    print("----------------------------training finished---------------------------")
    print("best iteration:", best_iteration)
    #     print(feature_importance_values)

    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # Overall validation score
    valid_auc = roc_auc_score(y, out_of_fold)

    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    if need_oof:
        return submission, feature_importances, metrics, out_of_fold
    else:
        return submission, feature_importances, metrics


def plot_feature_importances(df, top=15):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better.

    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance

    Returns:
        shows a plot of the 15 most importance features

        df (dataframe): feature importances sorted by importance (highest to lowest)
        with a column for normalized importance
        """

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 5*int(top/15)))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:top]))),
            df['importance_normalized'].head(top),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:top]))))
    ax.set_yticklabels(df['feature'].head(top))

    # Plot labeling
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances')
    plt.show()

    return df


