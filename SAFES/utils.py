import time
import pandas as pd

def df_to_Xy(df, y_label):
    return df.drop(y_label, axis=1), df[y_label]

def Xy_to_df(X, y):
    df = X.copy()
    df[y.name] = y
    return df

def convert_categorical_series_to_binary(series, one_categories):
    series_copy = series.copy()
    series_copy[:] = 0
    series_copy[series.isin(one_categories)] = 1
    # series.replace(one_categories, 1, inplace=True)
    # series.replace(series[series!=1].unique(), 0, inplace=True)
    return series_copy

def get_time_passed(start):
    hours, remainder = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)

def metric_uses_score(metric_name):
    uses_score_metrics = ['roc_auc_score']
    return metric_name in uses_score_metrics

def undo_aif360_dummies(df, non_dummy_cols, sep=None):
    return pd.concat((df[non_dummy_cols],
                      pd.from_dummies(df.drop(non_dummy_cols, axis=1), sep=sep)), 
                      axis=1)