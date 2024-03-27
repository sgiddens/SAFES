import pandas as pd
import json

COMPAS_URL = "https://github.com/propublica/compas-analysis/blob/"\
    "bafff5da3f2e45eca6c2d5055faad269defd135a/"\
    "compas-scores-two-years.csv?raw=true"

def preprocessing_pipeline(file_path="preprocessed_data/compas/", 
                           csv_name="compas_preprocessed.csv",
                           json_name="compas_domain.json"):
    df_compas = fetch_compas_dataset()
    df_compas = filter_bad_values(df_compas)
    df_compas = preprocess_all_variables(df_compas)
    save_compas_information(df_compas, file_path, csv_name, json_name)
    return df_compas

def fetch_compas_dataset():
    df_compas = pd.read_csv(COMPAS_URL, index_col='id')
    return df_compas

def filter_bad_values(df):
    # Filter as was done in original analysis
    df = df[
        (df.days_b_screening_arrest <= 30)
      & (df.days_b_screening_arrest >= -30)
      & (df.is_recid != -1)
      & (df.score_text != 'N/A')
    ]
    return df

def preprocess_all_variables(df):
    df = preprocess_sex(df)
    df = preprocess_age_cat(df)
    df = preprocess_race(df)
    df = preprocess_c_charge_degree(df)
    df = preprocess_priors_count(df)
    df = preprocess_two_year_recid(df)
    df = drop_remaining_variables(df)
    return df

def preprocess_sex(df):
    def sex_map(sex):
        if sex=="Male":
            return 1
        else:
            return 0
    df["sex-num"] = df["sex"].apply(sex_map)
    df = df.drop("sex", axis=1)
    return df

def preprocess_age_cat(df):
    def age_cat_map(age_cat):
        if age_cat=="Less than 25":
            return 0
        elif age_cat=="25 - 45":
            return 1
        else:
            return 2
    df["age_cat-num"] = df["age_cat"].apply(age_cat_map)
    df = df.drop("age_cat", axis=1)
    return df

def preprocess_race(df):
    # Only keep two races
    df = df[df.race.isin(['African-American', 'Caucasian'])]
    def race_map(race):
        if race=="Caucasian":
            return 1
        else:
            return 0
    df["race-num"] = df["race"].apply(race_map)
    df = df.drop("race", axis=1)
    return df

def preprocess_c_charge_degree(df):
    def c_charge_degree_map(c_charge_degree):
        if c_charge_degree=="F":
            return 1
        else:
            return 0
    df["c_charge_degree-num"] = df["c_charge_degree"].apply(c_charge_degree_map)
    df = df.drop("c_charge_degree", axis=1)
    return df

def preprocess_priors_count(df):
    def priors_count_reducer(priors_count):
        if priors_count>3:
            return 2
        elif priors_count>0:
            return 1
        else:
            return 0
    df["priors_count-reduced-num"] = df["priors_count"].apply(priors_count_reducer)
    df = df.drop("priors_count", axis=1)
    return df

def preprocess_two_year_recid(df):
    return df

def drop_remaining_variables(df):
    return df[["sex-num", "race-num", "age_cat-num",
               "c_charge_degree-num", 
               "priors_count-reduced-num", 
               "two_year_recid"]]

def save_compas_information(df, file_path, csv_name, json_name):
    df.to_csv(file_path + csv_name, index=True)

    # TODO: Incorporate building this dictionary into various functions
    compas_domain = {
        "sex-num": 2,
        "race-num": 2,
        "age_cat-num": 3,
        "c_charge_degree-num": 2,
        "priors_count-reduced-num": 3,
        "two_year_recid": 2,
    }

    with open(file_path + json_name, 'w') as json_file:
        json.dump(compas_domain, json_file)