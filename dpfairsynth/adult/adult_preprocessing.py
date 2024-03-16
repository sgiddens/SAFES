from ucimlrepo import fetch_ucirepo
import pandas as pd
import json
import numpy as np

from mbi import Dataset

def custom_aif360_conversion(df):
    df['age-decade'] = df['age-shifted'].apply(lambda x: x//10*10)
    return df

def custom_distortion(vold, vnew):
    def adjustInc(a):
        if a == "<=50K":
            return 0
        elif a == ">50K":
            return 1
        else:
            return int(a)

    # value that will be returned for events that should not occur
    bad_val = 3.0

    eOld = int(vold['education-reduced'])
    eNew = int(vnew['education-reduced'])

    # Education cannot be lowered or increased in more than 1 year
    if (eNew < eOld) | (eNew > eOld+1):
        return bad_val
    
    aOld = float(vold['age-decade'])
    aNew = float(vnew['age-decade'])

    # Age cannot be increased or decreased in more than a decade
    if np.abs(aOld-aNew) > 10.0:
        return bad_val

    # Penalty of 2 if age is decreased or increased
    if np.abs(aOld-aNew) > 0:
        return 2.0
    
    incOld = adjustInc(vold['income>50K'])
    incNew = adjustInc(vnew['income>50K'])

    # final penalty according to income
    if incOld > incNew:
        return 1.0
    else:
        return 0.0

def load_preprocessed_adult_data(file_path="preprocessed_data/adult/",
                                 csv_file="adult_preprocessed.csv",
                                 json_file="adult_domain.json",
                                 features_to_drop=[
                                    "capital-gain-1000s",
                                    "capital-loss-50s",
                                    "native-country-reduced",
                                    "relationship-factorized",
                                    "workclass-reduced",
                                    "hours-per-week",
                                    "occupation-reduced",
                                    "marital-status-reduced",
                                 ]):
    full_adult_data = Dataset.load(file_path+csv_file, file_path+json_file)
    df_subset_adult_data = full_adult_data.df.drop(features_to_drop, axis=1)
    domain_dict = {k: full_adult_data.domain.config[k] for k in df_subset_adult_data.columns}
    return df_subset_adult_data, domain_dict

def preprocessing_pipeline(file_path="preprocessed_data/adult/", 
                           csv_name="adult_preprocessed.csv",
                           json_name="adult_domain.json"):
    df_adult = fetch_adult_dataset()
    df_adult = preprocess_all_variables(df_adult)
    save_adult_information(df_adult, file_path, csv_name, json_name)
    return df_adult

def fetch_adult_dataset():
    UCI_adult = fetch_ucirepo(id=2) 
    df_adult = UCI_adult.data.original
    return df_adult

def preprocess_all_variables(df):
    df = preprocess_age(df)
    df = preprocess_workclass(df)
    df = preprocess_fnlwgt(df)
    df = preprocess_education(df)
    df = preprocess_marital_status(df)
    df = preprocess_occupation(df)
    df = preprocess_relationship(df)
    df = preprocess_race(df)
    df = preprocess_sex(df)
    df = preprocess_capital_gain(df)
    df = preprocess_capital_loss(df)
    df = preprocess_hours_per_week(df)
    df = preprocess_native_country(df)
    df = preprocess_income(df)
    return df

def preprocess_age(df):
    df["age-shifted"] = df["age"] - 17
    df = df.drop("age", axis=1)
    return df

def preprocess_workclass(df):
    def workclass_reducer(workclass):
        if workclass=="Private":
            return "Private"
        elif workclass=="Self-emp-not-inc":
            return "Self-emp-not-inc"
        elif workclass=="Self-emp-inc":
            return "Self-emp-inc"
        elif workclass=="Federal-gov":
            return "Federal-gov"
        elif workclass=="Local-gov" or workclass=="State-gov":
            return "Local-State-gov"
        else:
            return "Other"
    df["workclass-reduced"] = df["workclass"].apply(workclass_reducer)
    df = df.drop("workclass", axis=1)
    return df

def preprocess_fnlwgt(df):
    df = df.drop("fnlwgt", axis=1)
    return df

def preprocess_education(df):
    df = df.drop("education", axis=1)
    def education_num_reducer(education_num):
        if education_num<=6:
            return 0
        elif education_num>=14:
            return 8
        else:
            return int(education_num)-6
#         if education_num<9: # < HS
#             return 1
#         elif education_num==9 or education_num==10: # HS or some college
#             return 2
#         elif education_num==11 or education_num==12: # Associates degree
#             return 3
#         elif education_num==13: # Bachelor's degree
#             return 4
#         elif education_num==14: # Masters degree
#             return 5
#         else: # Professional or doctorate degree
#             return 6
    df["education-reduced"] = df["education-num"].apply(education_num_reducer)
    df = df.drop("education-num", axis=1)
    return df

def preprocess_marital_status(df):
    def marital_status_reducer(marital_status):
        if marital_status=="Married-civ-spouse" or marital_status=="Married-AF-spouse": # Married
            return 1
        else: # Not married or separated
            return 0
    
    df["marital-status-reduced"] = df["marital-status"].apply(marital_status_reducer)
    df = df.drop("marital-status", axis=1)
    return df

def preprocess_occupation(df):
    def occupation_reducer(occupation):
        if pd.isna(occupation):
            return "?"
        else:
            return occupation
    
    df["occupation-reduced"] = df["occupation"].apply(occupation_reducer)
    df = df.drop("occupation", axis=1)
    return df

def preprocess_relationship(df):
#     # Leave as is but move columns
#     df = df[
#         [col for col in df.columns if col!="relationship"] + 
#         ["relationship"]
#     ]
    df["relationship-factorized"] = pd.factorize(df["relationship"])[0]
    df = df.drop("relationship", axis=1)
    return df

def preprocess_race(df):
    def race_reducer(race):
        if race=="White": # Privileged
            return 1
        else: # Non-privileged
            return 0
    df["race-reduced"] = df["race"].apply(race_reducer)
    df = df.drop("race", axis=1)
    return df

def preprocess_sex(df):
    def sex_reducer(sex):
        if sex=="Male":
            return 1
        else:
            return 0
    df["sex-num"] = df["sex"].apply(sex_reducer)
    df = df.drop("sex", axis=1)
    return df

def preprocess_capital_gain(df):
    df["capital-gain-1000s"] = df["capital-gain"]//1000
    df = df.drop("capital-gain", axis=1)
    return df

def preprocess_capital_loss(df):
    df["capital-loss-50s"] = df["capital-loss"]//50
    df = df.drop("capital-loss", axis=1)
    return df

def preprocess_hours_per_week(df):
    # Leave as is but move columns
    df = df[
        [col for col in df.columns if col!="hours-per-week"] + 
        ["hours-per-week"]
    ]
    return df

def preprocess_native_country(df):
    def native_country_reducer(native_country):
        if pd.isna(native_country):
            return "?"
        else:
            return native_country
    df["native-country-reduced"] = df["native-country"].apply(native_country_reducer)
    df = df.drop("native-country", axis=1)
    return df

def preprocess_income(df):
    def income_reducer(income):
        if income=="<=50K" or income=="<=50K.":
            return 0
        else:
            return 1
    df["income>50K"] = df["income"].apply(income_reducer)
    df = df.drop("income", axis=1)
    return df

def save_adult_information(df, file_path, csv_name, json_name):
    # Save resulting files
    df.to_csv(file_path + csv_name, index=False)

    # TODO: Incorporate building this dictionary into various functions
    adult_domain = {
        "age-shifted": 80,
        "workclass-reduced": 6,
        "education-reduced": 9,
        "marital-status-reduced": 2,
        "occupation-reduced": 15,
        "relationship-factorized": 6,
        "race-reduced": 2,
        "sex-num": 2,
        "capital-gain-1000s": 100,
        "capital-loss-50s": 100,
        "hours-per-week": 99,
        "native-country-reduced": 42,
        "income>50K": 2
    }
        
    with open(file_path + json_name, 'w') as json_file:
        json.dump(adult_domain, json_file)