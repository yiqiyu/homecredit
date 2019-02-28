import pandas as pd
import numpy as np
import gc
import importlib

import matplotlib.pyplot as plt
import seaborn as sns


def get_age_label(days_birth):
    """ Return the age group label (int). """
    age_years = -days_birth / 365
    if age_years < 27: return 1
    elif age_years < 40: return 2
    elif age_years < 50: return 3
    elif age_years < 65: return 4
    elif age_years < 99: return 5
    else: return 0


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

    # for df in [app_train, app_test]:
    #     df['BUREAU_INCOME_CREDIT_RATIO'] = df['BURO_AMT_CREDIT_SUM_MEAN'] / df['AMT_INCOME_TOTAL']
    #     df['BUREAU_ACTIVE_CREDIT_TO_INCOME_RATIO'] = df['ACTIVE_AMT_CREDIT_SUM_SUM'] / df['AMT_INCOME_TOTAL']
    #
    #     df['CURRENT_TO_APPROVED_CREDIT_MIN_RATIO'] = df['APPROVED_AMT_CREDIT_MIN'] / df['AMT_CREDIT']
    #     df['CURRENT_TO_APPROVED_CREDIT_MAX_RATIO'] = df['APPROVED_AMT_CREDIT_MAX'] / df['AMT_CREDIT']
    #     df['CURRENT_TO_APPROVED_CREDIT_MEAN_RATIO'] = df['APPROVED_AMT_CREDIT_MEAN'] / df['AMT_CREDIT']
    #
    #     df['CURRENT_TO_APPROVED_ANNUITY_MAX_RATIO'] = df['APPROVED_AMT_ANNUITY_MAX'] / df['AMT_ANNUITY']
    #     df['CURRENT_TO_APPROVED_ANNUITY_MEAN_RATIO'] = df['APPROVED_AMT_ANNUITY_MEAN'] / df['AMT_ANNUITY']
    #     df['PAYMENT_MIN_TO_ANNUITY_RATIO'] = df['INSTAL_AMT_PAYMENT_MIN'] / df['AMT_ANNUITY']
    #     df['PAYMENT_MAX_TO_ANNUITY_RATIO'] = df['INSTAL_AMT_PAYMENT_MAX'] / df['AMT_ANNUITY']
    #     df['PAYMENT_MEAN_TO_ANNUITY_RATIO'] = df['INSTAL_AMT_PAYMENT_MEAN'] / df['AMT_ANNUITY']
    #     # PREVIOUS TO CURRENT CREDIT TO ANNUITY RATIO
    #     df['CTA_CREDIT_TO_ANNUITY_MAX_RATIO'] = df['APPROVED_CREDIT_TO_ANNUITY_RATIO_MAX'] / df[
    #         'CREDIT_TO_ANNUITY_RATIO']
    #     df['CTA_CREDIT_TO_ANNUITY_MEAN_RATIO'] = df['APPROVED_CREDIT_TO_ANNUITY_RATIO_MEAN'] / df[
    #         'CREDIT_TO_ANNUITY_RATIO']
    #     # DAYS DIFFERENCES AND RATIOS
    #     df['DAYS_DECISION_MEAN_TO_BIRTH'] = df['APPROVED_DAYS_DECISION_MEAN'] / df['DAYS_BIRTH']
    #     df['DAYS_CREDIT_MEAN_TO_BIRTH'] = df['BURO_DAYS_CREDIT_MEAN'] / df['DAYS_BIRTH']
    #     df['DAYS_DECISION_MEAN_TO_EMPLOYED'] = df['APPROVED_DAYS_DECISION_MEAN'] / df['DAYS_EMPLOYED']
    #     df['DAYS_CREDIT_MEAN_TO_EMPLOYED'] = df['BURO_DAYS_CREDIT_MEAN'] / df['DAYS_EMPLOYED']

    return app_train, app_test


def load_extra_features(app_train, app_test, *tables):
    docs = [_f for _f in app_train.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in app_train.columns if (_f.startswith('FLAG_')) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    for df in [app_train, app_test]:
        df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
        df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
        df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
        df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
        df['NEW_PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
        df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['WORKING_LIFE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['INCOME_PER_FAM'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
        df['document_sum'] = df['FLAG_DOCUMENT_2'] + df['FLAG_DOCUMENT_3'] + df['FLAG_DOCUMENT_4'] + df[
            'FLAG_DOCUMENT_5'] + df['FLAG_DOCUMENT_6'] + df['FLAG_DOCUMENT_7'] + df['FLAG_DOCUMENT_8'] + df[
                                  'FLAG_DOCUMENT_9'] + df['FLAG_DOCUMENT_10'] + df['FLAG_DOCUMENT_11'] + df[
                                  'FLAG_DOCUMENT_12'] + df['FLAG_DOCUMENT_13'] + df['FLAG_DOCUMENT_14'] + df[
                                  'FLAG_DOCUMENT_15'] + df['FLAG_DOCUMENT_16'] + df['FLAG_DOCUMENT_17'] + df[
                                  'FLAG_DOCUMENT_18'] + df['FLAG_DOCUMENT_19'] + df['FLAG_DOCUMENT_20'] + df[
                                  'FLAG_DOCUMENT_21']
        df['credit_minus_goods'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
        df['reg_div_publish'] = df['DAYS_REGISTRATION'] / df['DAYS_ID_PUBLISH']
        df['birth_div_reg'] = df['DAYS_BIRTH'] / df['DAYS_REGISTRATION']
        df['ANN_LENGTH_EMPLOYED_RATIO'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY']) / df['DAYS_EMPLOYED']
        df['age_finish'] = df['DAYS_BIRTH'] * (-1.0 / 365) + (df['AMT_CREDIT'] / df['AMT_ANNUITY']) * (
                1.0 / 12)  # how old when finish
        df['ANN_LENGTH_AGE_RATIO'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY']) * (1.0 / 12) / df['DAYS_BIRTH'] * (
                -1.0 / 365)
        df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
        df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)

        df["OBS_30_60_diff"] = (df["OBS_30_CNT_SOCIAL_CIRCLE"] - df["OBS_60_CNT_SOCIAL_CIRCLE"])

        df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_label(x))
        df['AMT_MISSING_FIELDS'] = df.isnull().sum(axis=1).values
        df['NEW_CREDIT_TO_GOODS_DIFF'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']


    for table in tables:
        app_train = app_train.merge(right=table.reset_index(), how='left', on='SK_ID_CURR')
        app_test = app_test.merge(right=table.reset_index(), how='left', on='SK_ID_CURR')
    return app_train, app_test


def load_bureau(bureau, buro_balance):
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    bb_cat = [col for col in buro_balance.columns if len(buro_balance[col].unique()) == 2]
    bb_aggregations.update({col: ["mean"] for col in bb_cat})
    buro_balance_agg = buro_balance.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    buro_balance_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in buro_balance_agg.columns.tolist()])
    bureau = bureau.join(buro_balance_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del buro_balance_agg
    gc.collect()

    bureau['STATUS_12345'] = 0
    for i in range(1, 6):
        bureau['STATUS_12345'] += bureau['STATUS_{}'.format(i)]

    bureau["DAYS_CREDIT_ITERVAL"] = bureau["DAYS_CREDIT"] - bureau["DAYS_CREDIT_ENDDATE"]
    bureau["CREDIT_SUM_DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM"]
    bureau["CREDIT_SUM_OVERDUE_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / bureau["AMT_CREDIT_SUM"]
    bureau["CREDIT_SUM_LIMIT_DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM_LIMIT"]
    bureau["CREDIT_SUM_LIMIT_OVERDUE_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / bureau["AMT_CREDIT_SUM_LIMIT"]
    bureau["ANNUITY_DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_ANNUITY"]
    bureau["ANNUITY_OVERDUE_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / bureau["AMT_ANNUITY"]
    bureau["MAX_OVERDUE_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / bureau["AMT_CREDIT_MAX_OVERDUE"]

    bureau['ENDDATE_DIF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    num_aggregations = {
        'STATUS_0': ['mean'],
        'STATUS_1': ['mean'],
        'STATUS_12345': ['mean'],
        'STATUS_C': ['mean'],
        'STATUS_X': ['mean'],
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],

        "DAYS_CREDIT_ITERVAL": ['min', 'max', 'mean', 'var'],
        "CREDIT_SUM_DEBT_RATIO": ['min', 'max', 'mean', 'var'],
        "CREDIT_SUM_OVERDUE_RATIO": ['min', 'max', 'mean', 'var'],
        "CREDIT_SUM_LIMIT_DEBT_RATIO": ['min', 'max', 'mean', 'var'],
        "CREDIT_SUM_LIMIT_OVERDUE_RATIO": ['min', 'max', 'mean', 'var'],
        "ANNUITY_DEBT_RATIO": ['min', 'max', 'mean', 'var'],
        "ANNUITY_OVERDUE_RATIO": ['min', 'max', 'mean', 'var'],
        "MAX_OVERDUE_RATIO": ['min', 'max', 'mean', 'var'],
        'DEBT_CREDIT_DIFF': ['mean', 'sum'],
        'ENDDATE_DIF': ['mean', 'sum', "max"],
    }
    cat_aggregations = {col: ["mean"] for col in bureau.columns if len(bureau[col].unique()) == 2}
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    BUREAU_ACTIVE_AGG = {
        'DAYS_CREDIT': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['min', 'mean'],
        'DEBT_PERCENTAGE': ['mean'],
        'DEBT_CREDIT_DIFF': ['mean'],
        'CREDIT_TO_ANNUITY_RATIO': ['mean'],
        'MONTHS_BALANCE_MEAN': ['mean', 'var'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
    }

    BUREAU_CLOSED_AGG = {
        'DAYS_CREDIT': ['max', 'var'],
        'DAYS_CREDIT_ENDDATE': ['max'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'sum'],
        'DAYS_CREDIT_UPDATE': ['max'],
        'ENDDATE_DIF': ['mean'],
        'STATUS_12345': ['mean'],
    }

    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(BUREAU_ACTIVE_AGG)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(BUREAU_CLOSED_AGG)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg
    gc.collect()
    return bureau_agg


def load_prev(prev):
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev["APP_DURATION"] = prev["DAYS_TERMINATION"] - prev["DAYS_DECISION"]
    prev["APP_PAY_DURATION"] = prev["DAYS_LAST_DUE"] - prev["DAYS_FIRST_DRAWING"]
    prev["INTREST_RATE_RATIO"] = prev["RATE_INTEREST_PRIVILEGED"] / prev["RATE_INTEREST_PRIMARY"]
    prev['CREDIT_TO_GOODS_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_GOODS_PRICE']

    total_payment = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    prev['SIMPLE_INTERESTS'] = (total_payment / prev['AMT_CREDIT'] - 1) / prev['CNT_PAYMENT']

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],

        "APP_DURATION": ['max', 'mean', 'sum'],
        "APP_PAY_DURATION": ['max', 'mean', 'sum'],
        "INTREST_RATE_RATIO": ['max', 'mean', 'sum', 'var'],
        'CREDIT_TO_GOODS_RATIO': ['max', 'mean', 'sum', 'var'],
    }
    cat_aggregations = {col: ["mean"] for col in prev.columns if len(prev[col].unique()) == 2}
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    #     print(prev_agg.columns.tolist())
    #     prev_agg = prev_agg.join(prev_agg, how='left', on='SK_ID_CURR')

    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    del approved, approved_agg
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg
    gc.collect()
    return prev_agg


def load_install(inst):
    inst["DPD"] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst["DBD"] = inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']
    inst['DPD'] = inst['DPD'].apply(lambda x: x if x > 0 else 0)
    inst['DBD'] = inst['DBD'].apply(lambda x: x if x > 0 else 0)
    inst['PAYMENT_PERC'] = inst['AMT_PAYMENT'] / inst['AMT_INSTALMENT']
    inst['PAYMENT_DIFF'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']

    inst['LATE_PAYMENT'] = inst['DBD'].apply(lambda x: 1 if x > 0 else 0)
    inst['LATE_PAYMENT_RATIO'] = inst.apply(lambda x: x['INSTALMENT_PAYMENT_RATIO'] if x['LATE_PAYMENT'] == 1 else 0,
                                          axis=1)
    inst['SIGNIFICANT_LATE_PAYMENT'] = inst['LATE_PAYMENT_RATIO'].apply(lambda x: 1 if x > 0.05 else 0)
    # Flag k threshold late payments
    inst['DPD_7'] = inst['DPD'].apply(lambda x: 1 if x >= 7 else 0)
    inst['DPD_15'] = inst['DPD'].apply(lambda x: 1 if x >= 15 else 0)

    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],

        'SK_ID_PREV': ['size', 'nunique'],
        'LATE_PAYMENT': ['mean', 'sum'],
        'SIGNIFICANT_LATE_PAYMENT': ['mean', 'sum'],
        'LATE_PAYMENT_RATIO': ['mean'],
        'DPD_7': ['mean'],
        'DPD_15': ['mean'],
    }
    aggregations.update({col: ["mean"] for col in inst.columns if len(inst[col].unique()) == 2})
    inst_agg = inst.groupby('SK_ID_CURR').agg(aggregations)
    #     print(inst_agg.columns)
    inst_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in inst_agg.columns.tolist()])
    inst_agg["INSTAL_TOTAL_PAYMENT_RATE"] = inst_agg["INSTAL_AMT_PAYMENT_SUM"] / inst_agg["INSTAL_AMT_INSTALMENT_SUM"]
    inst_agg['INSTAL_COUNT'] = inst.groupby('SK_ID_CURR').size()
    #     del inst
    #     gc.collect()
    return inst_agg


def load_cash(cash):
    cash["INSTALMENT_FUTURE_RATIO"] = cash["CNT_INSTALMENT_FUTURE"] / cash["CNT_INSTALMENT"]
    cash["INSTALMENT_BALANCE_RATIO"] = cash["CNT_INSTALMENT_FUTURE"] / cash["MONTHS_BALANCE"].abs()
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        "CNT_INSTALMENT": ['max', "first", "last", 'mean', 'sum', 'var'],
        "CNT_INSTALMENT_FUTURE": ['max', 'mean', 'sum', 'var', "last"],

        "INSTALMENT_FUTURE_RATIO": ['max', 'mean', 'sum', 'var'],
        "INSTALMENT_BALANCE_RATIO": ['max', 'mean', 'sum', 'var'],
        "NAME_CONTRACT_STATUS_Completed": ["mean"]
    }
    aggregations.update({col: ["mean"] for col in cash.columns if len(cash[col].unique()) == 2})
    cash_agg = cash.groupby('SK_ID_CURR').agg(aggregations)
    cash_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in cash_agg.columns.tolist()])
    # Count pos cash accounts
    cash_agg['POS_COUNT'] = cash.groupby('SK_ID_CURR').size()

    cash_agg["POS_COMPLETED_BEFORE_MEAN"] = cash_agg['POS_CNT_INSTALMENT_first'] - cash_agg['POS_CNT_INSTALMENT_last']
    cash_agg['POS_COMPLETED_BEFORE_MEAN'] = cash_agg.apply(lambda x: 1 if x['POS_COMPLETED_BEFORE_MEAN'] > 0
                                                              and x['NAME_CONTRACT_STATUS_Completed_mean'] > 0 else 0, axis=1)
    cash_agg['POS_REMAINING_INSTALMENTS_RATIO'] = cash_agg['CNT_INSTALMENT_FUTURE_last'] / cash_agg['CNT_INSTALMENT_last']

    return cash_agg


def load_credit(credit):
    credit["INSTALMENT_MATURE_BALANCE_RATIO"] = credit["CNT_INSTALMENT_MATURE_CUM"] / credit["MONTHS_BALANCE"].abs()
    credit["POS_AMT_MEAN"] = credit["AMT_DRAWINGS_POS_CURRENT"] / credit["CNT_DRAWINGS_POS_CURRENT"]
    credit["PAYMENT_DIFF"] = credit["AMT_PAYMENT_CURRENT"] - credit["AMT_PAYMENT_TOTAL_CURRENT"]
    credit["RECEIVABLE_DIFF"] = credit["AMT_TOTAL_RECEIVABLE"] - credit["AMT_RECIVABLE"]
    credit["RECEIVABLE_PRINCIPAL_DIFF"] = credit["AMT_TOTAL_RECEIVABLE"] - credit["AMT_RECEIVABLE_PRINCIPAL"]

    credit['LIMIT_USE'] = credit['AMT_BALANCE'] / credit['AMT_CREDIT_LIMIT_ACTUAL']
    credit['PAYMENT_DIV_MIN'] = credit['AMT_PAYMENT_CURRENT'] / credit['AMT_INST_MIN_REGULARITY']
    credit['LATE_PAYMENT'] = credit['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    credit['DRAWING_LIMIT_RATIO'] = credit['AMT_DRAWINGS_ATM_CURRENT'] / credit['AMT_CREDIT_LIMIT_ACTUAL']

    for col in ["CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_OTHER_CURRENT", "CNT_DRAWINGS_POS_CURRENT"]:
        credit["RELATIVE" + col[col.find("_"):]] = credit[col] / credit["CNT_DRAWINGS_CURRENT"]
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', "min"],
        'AMT_BALANCE': ['max', 'mean', 'sum', 'var'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean', 'var'],
        "AMT_DRAWINGS_ATM_CURRENT": ['max', 'mean', 'sum', 'var'],
        "AMT_DRAWINGS_CURRENT": ['max', 'mean', 'sum', 'var'],
        "AMT_DRAWINGS_OTHER_CURRENT": ['max', 'mean', 'sum', 'var'],
        "AMT_DRAWINGS_POS_CURRENT": ['max', 'mean', 'sum', 'var'],
        "AMT_INST_MIN_REGULARITY": ['max', 'mean', 'sum', 'var'],
        "AMT_PAYMENT_CURRENT": ['max', 'mean', 'sum', 'var'],
        "AMT_PAYMENT_TOTAL_CURRENT": ['max', 'mean', 'sum', 'var'],
        "AMT_RECEIVABLE_PRINCIPAL": ['max', 'mean', 'sum', 'var'],
        "AMT_RECIVABLE": ['max', 'mean', 'sum', 'var'],
        "AMT_TOTAL_RECEIVABLE": ['max', 'mean', 'var'],
        "CNT_DRAWINGS_ATM_CURRENT": ['max', 'mean', 'sum', 'var'],
        "CNT_DRAWINGS_CURRENT": ['max', 'mean', 'sum', 'var'],
        "CNT_DRAWINGS_OTHER_CURRENT": ['max', 'mean', 'sum', 'var'],
        "CNT_DRAWINGS_POS_CURRENT": ['max', 'mean', 'sum', 'var'],
        "CNT_INSTALMENT_MATURE_CUM": ['max', 'mean', 'sum', 'var'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],

        "INSTALMENT_MATURE_BALANCE_RATIO": ['max', 'mean', 'var'],
        "POS_AMT_MEAN": ['max', 'mean', 'sum', 'var'],
        "PAYMENT_DIFF": ['max', 'mean', 'sum', 'var'],
        "RECEIVABLE_DIFF": ['max', 'mean', 'sum', 'var'],
        "RECEIVABLE_PRINCIPAL_DIFF": ['max', 'mean', 'sum', 'var'],
        'LIMIT_USE': ['max', 'mean'],
        'PAYMENT_DIV_MIN': ['min', 'mean'],
        'LATE_PAYMENT': ['max', 'sum'],
    }
    aggregations.update({"RELATIVE" + col[col.find("_"):]: ['max', 'mean', 'sum', 'var'] for col in
                         ["CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_OTHER_CURRENT", "CNT_DRAWINGS_POS_CURRENT"]})
    aggregations.update({col: ["mean"] for col in credit.columns if len(credit[col].unique()) == 2})
    cc_agg = credit.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = credit.groupby('SK_ID_CURR').size()
    return cc_agg


def load_all_tables(train="app_train_new.csv", test="app_test_new.csv"):
    to_del_cols = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
                   'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
                   'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
                   'AMT_GOODS_PRICE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MODE',
                   'ELEVATORS_MODE', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MODE', 'LANDAREA_MODE',
                   'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE']
    app_train = load_dataframe(train)
    app_test = load_dataframe(test)
    app_train, app_test = load_extra_features(app_train, app_test)
    # app_train = app_train.drop(to_del_cols, axis=1)
    # app_test = app_test.drop(to_del_cols, axis=1)


    bureau = load_dataframe("bureau_new.csv")
    buro_balance = load_dataframe("bureau_balance_new.csv")
    bbb = load_bureau(bureau, buro_balance)

    app_train = app_train.merge(right=bbb.reset_index(), how='left', on='SK_ID_CURR')
    app_test = app_test.merge(right=bbb.reset_index(), how='left', on='SK_ID_CURR')
    del bureau, buro_balance
    gc.collect()

    for name, load_method in {"cash": load_cash, "credit": load_credit, "installments": load_install,
                              "previous": load_prev}.items():
        df = load_method(load_dataframe(name + "_new.csv"))
        app_train = app_train.merge(right=df.reset_index(), how='left', on='SK_ID_CURR')
        app_test = app_test.merge(right=df.reset_index(), how='left', on='SK_ID_CURR')
        del df
        gc.collect()

    to_del = del_useless_cols(app_train)
    app_test = app_test.drop(to_del, axis=1)

    return app_train, app_test


def del_useless_cols(ds):
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
