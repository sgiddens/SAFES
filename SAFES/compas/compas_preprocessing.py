import pandas as pd
import json

from mbi import Domain

COMPAS_URL = "https://github.com/propublica/compas-analysis/blob/"\
    "bafff5da3f2e45eca6c2d5055faad269defd135a/"\
    "compas-scores-two-years.csv?raw=true"

def custom_distortion(vold, vnew):
    # Distortion cost
    distort = {}
    distort['two_year_recid'] = pd.DataFrame(
                                {0:     [0., 2.],
                                1:     [2., 0.]},
                                index=[0, 1])
    distort['age_cat-num'] = pd.DataFrame(
                            {0: [0., 1., 2.],
                            1:  [1., 0., 1.],
                            2:  [2., 1., 0.]},
                            index=[0, 1, 2])
    distort['c_charge_degree-num'] = pd.DataFrame(
                            {0:   [0., 2.],
                            1:    [1., 0.]},
                            index=[0, 1])
    distort['priors_count-reduced-num'] = pd.DataFrame(
                            {0: [0., 1., 2.],
                            1:  [1., 0., 1.],
                            2:  [2., 1., 0.]},
                            index=[0, 1, 2])
    distort['sex-num'] = pd.DataFrame(
                        {1:    [0., 2.],
                         0:    [2., 0.]},
                         index=[1, 0])
    distort['race-reduced-num'] = pd.DataFrame(
                        {0:    [0., 2.],
                         1:    [2., 0.]},
                         index=[0, 1])
    total_cost = 0.0
    for k in vold:
        if k in vnew:
            total_cost += distort[k].loc[int(vnew[k]), int(vold[k])]
    return total_cost

def load_preprocessed_compas_data(file_path="preprocessed_data/compas/",
                                 csv_file="compas_preprocessed.csv",
                                 json_file="compas_domain.json",
                                 features_to_drop=[]):
    full_compas_data = pd.read_csv(file_path+csv_file, index_col='id')
    df_subset_compas_data = full_compas_data.drop(features_to_drop, axis=1)
    config = json.load(open(file_path+json_file))
    domain = Domain(config.keys(), config.values())
    domain_dict = {k: domain.config[k] for k in df_subset_compas_data.columns}
    return df_subset_compas_data, domain_dict

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
        if sex=="Female":
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
    df["race-reduced-num"] = df["race"].apply(race_map)
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
    return df[["sex-num", "race-reduced-num", "age_cat-num",
               "c_charge_degree-num", 
               "priors_count-reduced-num", 
               "two_year_recid"]]

def save_compas_information(df, file_path, csv_name, json_name):
    df.to_csv(file_path + csv_name, index=True)

    # TODO: Incorporate building this dictionary into various functions
    compas_domain = {
        "sex-num": 2,
        "race-reduced-num": 2,
        "age_cat-num": 3,
        "c_charge_degree-num": 2,
        "priors_count-reduced-num": 3,
        "two_year_recid": 2,
    }

    with open(file_path + json_name, 'w') as json_file:
        json.dump(compas_domain, json_file)