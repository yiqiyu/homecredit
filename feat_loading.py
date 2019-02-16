import pandas as pd
import numpy as np
import gc
import importlib

import matplotlib.pyplot as plt
import seaborn as sns


def load_dataframe(loc):
    df = pd.read_csv(loc).reset_index(drop = True)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    return df


def load_extra_feats_post(app_train, app_test):
    tables = []
    ori_train = load_dataframe("app_train_new.csv")
    ori_test = load_dataframe("app_test_new.csv")
    new_train = pd.DataFrame(app_train[["SK_ID_CURR"]])
    new_test = pd.DataFrame(app_test[["SK_ID_CURR"]])
    docs = [_f for _f in app_train.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in app_train.columns if (_f.startswith('FLAG_')) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    for df, new in [(ori_train, new_train), (ori_test, new_test)]:
        new['document_sum'] = df['FLAG_DOCUMENT_2'] + df['FLAG_DOCUMENT_3'] + df['FLAG_DOCUMENT_4'] + df[
            'FLAG_DOCUMENT_5'] + df['FLAG_DOCUMENT_6'] + df['FLAG_DOCUMENT_7'] + df['FLAG_DOCUMENT_8'] + df[
                                  'FLAG_DOCUMENT_9'] + df['FLAG_DOCUMENT_10'] + df['FLAG_DOCUMENT_11'] + df[
                                  'FLAG_DOCUMENT_12'] + df['FLAG_DOCUMENT_13'] + df['FLAG_DOCUMENT_14'] + df[
                                  'FLAG_DOCUMENT_15'] + df['FLAG_DOCUMENT_16'] + df['FLAG_DOCUMENT_17'] + df[
                                  'FLAG_DOCUMENT_18'] + df['FLAG_DOCUMENT_19'] + df['FLAG_DOCUMENT_20'] + df[
                                  'FLAG_DOCUMENT_21']
        new['credit_minus_goods'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
        new['reg_div_publish'] = df['DAYS_REGISTRATION'] / df['DAYS_ID_PUBLISH']
        new['birth_div_reg'] = df['DAYS_BIRTH'] / df['DAYS_REGISTRATION']
        new['ANN_LENGTH_EMPLOYED_RATIO'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY']) / df['DAYS_EMPLOYED']
        new['age_finish'] = df['DAYS_BIRTH'] * (-1.0 / 365) + (df['AMT_CREDIT'] / df['AMT_ANNUITY']) * (
                    1.0 / 12)  # how old when finish
        new['ANN_LENGTH_AGE_RATIO'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY']) * (1.0 / 12) / df['DAYS_BIRTH'] * (
                    -1.0 / 365)
        new['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
        new['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)

    app_train = app_train.merge(right=new_train.reset_index(), how='left', on='SK_ID_CURR')
    app_test = app_test.merge(right=new_test.reset_index(), how='left', on='SK_ID_CURR')
    del ori_train, ori_test
    gc.collect()

    return app_train, app_test


def del_single_variance_and_nan_too_much(ds):
    total_rows = ds.shape[0]
    null_count = 0
    to_del = []
    for col in ds.columns:
        uv = ds[col].unique()
        if ds[col].isnull().sum() / total_rows > 0.97:
            print(col)
            to_del.append(col)
            null_count += 1
        elif len(list(uv)) <= 1:
            print("%s has %s" % (col, uv))
            to_del.append(col)
            null_count += 1

    if to_del:
        ds.drop(columns=to_del, inplace=True)

    print("%s columns need to be dropped" % (null_count))
    return to_del
