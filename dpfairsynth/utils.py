def df_to_Xy(df, y_label):
    return df.drop(y_label, axis=1), df[y_label]

def Xy_to_df(X, y):
    df = X.copy()
    df[y.name] = y
    return df

def convert_categorical_series_to_binary(series, one_categories):
    series.replace(one_categories, 1, inplace=True)
    series[series!=1] = 0
    return series