import time
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score#, roc_auc_score
from aif360.sklearn.metrics import statistical_parity_difference, average_odds_difference

from custom_metrics import mean_outcome_difference, KS_test, roc_auc_score
from synthesizer import DataSynthesizer
import adult.adult_preprocessing as adult_preprocessing
import adult.adult_synthesizing as adult_synthesizing
import compas.compas_preprocessing as compas_preprocessing
import compas.compas_synthesizing as compas_synthesizing
import utils

class DPFairEvaluator():
    def __init__(self, dataset,
                 models=[
                     LogisticRegression(),
                 ],
                 dataset_utility_metrics=[
                     KS_test,
                 ],
                 dataset_fairness_metrics=[
                     mean_outcome_difference,
                 ],
                 model_utility_metrics=[
                     accuracy_score,
                     f1_score,
                     precision_score,
                     recall_score,
                     roc_auc_score,
                 ],
                 model_fairness_metrics=[
                     statistical_parity_difference,
                     average_odds_difference,
                 ],
                 random_state=50,
                 warm_start_file=None,
                 ):
        # Load and store train/test dataframes
        if dataset=="adult":
            self.dataset = dataset
            df, domain_dict = adult_preprocessing.load_preprocessed_adult_data()
            # Define settings
            (self.DP_settings_dict, 
             self.fair_settings_dict, 
             self.misc_settings_dict) = adult_synthesizing.define_settings()
            self.DP_settings_dict["domain_dict"] = domain_dict
        elif dataset=="compas":
            self.dataset = dataset
            df, domain_dict = compas_preprocessing.load_preprocessed_compas_data()
            # Define settings
            (self.DP_settings_dict,
             self.fair_settings_dict,
             self.misc_settings_dict) = compas_synthesizing.define_settings()
            self.DP_settings_dict["domain_dict"] = domain_dict
        else:
            raise ValueError("dataset value not currently supported.")
        self.y_label = self.fair_settings_dict["y_label"]
        self.favorable_classes = self.fair_settings_dict["favorable_classes"]

        self.protected_attribute_names_eval = self.misc_settings_dict["protected_attribute_names_eval"]
        self.privileged_classes_eval = self.misc_settings_dict["privileged_classes_eval"]
        self.comb_prot_attr_cols = [col for col in self.protected_attribute_names_eval
                                    if '*' in col]

        X, y = utils.df_to_Xy(df, self.y_label)
        (X_train, X_test, 
         y_train, y_test) = train_test_split(X, y, random_state=random_state)
        self.df_train = utils.Xy_to_df(X_train, y_train)
        self.df_test = utils.Xy_to_df(X_test, y_test)
        self.df_train_aif360_formatted = self.format_df_aif360(self.df_train.copy())
        self.df_test_aif360_formatted = self.format_df_aif360(self.df_test.copy())

        self.models = models

        self.dataset_fairness_metrics = dataset_fairness_metrics
        self.dataset_utility_metrics = dataset_utility_metrics
        self.model_fairness_metrics = model_fairness_metrics
        self.model_utility_metrics = model_utility_metrics

        # Initialize results dictionary
        if warm_start_file:
            self.load_results_dict(warm_start_file)
        else:
            self.reset_results_dict()

    def load_results_dict(self, warm_start_file):
        df = pd.read_csv(warm_start_file)
        self.results_dict = df.to_dict(orient='list')

    def reset_results_dict(self):
        results_dict = {
            "Linear epsilon (privacy)": [],
            "Epsilon (privacy)": [],
            "Epsilon (fairness)": [],
            "Dataset": [],
            "Model": [],
        }
        for metric in self.dataset_utility_metrics:
            if metric.__name__=="KS_test":
                results_dict.update({"KS Statistic": [],
                                     "KS p-value": []})
            else:
                results_dict.update({f"{metric.__name__}": []})

        for metric in self.dataset_fairness_metrics:
            if metric.__name__=="mean_outcome_difference":
                for prot_attr in self.protected_attribute_names_eval:
                    results_dict.update({f"{metric.__name__} "\
                                         f"({prot_attr})": []})
            else:
                results_dict.update({f"{metric.__name__}": []})

        for metric in self.model_utility_metrics:
            results_dict.update({f"{metric.__name__} (overall)": []})
            results_dict.update({f"{metric.__name__} (privileged {prot_attr})": []
                                 for prot_attr in self.protected_attribute_names_eval})
            results_dict.update({f"{metric.__name__} (unprivileged {prot_attr})": []
                                 for prot_attr in self.protected_attribute_names_eval})
            results_dict.update({f"{metric.__name__} (difference {prot_attr})": []
                                 for prot_attr in self.protected_attribute_names_eval})
            
        for metric in self.model_fairness_metrics:
            results_dict.update({f"{metric.__name__} ({prot_attr})": []
                                 for prot_attr in self.protected_attribute_names_eval})
        self.results_dict = results_dict

    def update_results_dict(self, new_results):
        if set(self.results_dict.keys())!=set(new_results.keys()):
            print("Keys in new results dictionary don't match full results "\
                  "dictionary. Full results dictionary was not updated.")
            return
        for k in self.results_dict.keys():
            self.results_dict[k].append(new_results[k])

    def save_results(self, save_path):
        df = pd.DataFrame(self.results_dict)
        filename = f"simulations_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        df.to_csv(save_path + filename, index=False)

    def create_comb_prot_attr_columns(self, df):
        if self.comb_prot_attr_cols:
            for col in self.comb_prot_attr_cols:
                # Separate into individual attributes
                prot_attrs = col.split('*')
                # Get privileged classes corresponding to individual attributes
                priv_classes = [self.privileged_classes_eval[
                    self.protected_attribute_names_eval.index(attr)] 
                    for attr in prot_attrs]
                # Create combined column
                df[col] = np.prod([np.where(
                    df[attr].isin(priv_class), 1, 0) 
                    for attr, priv_class in zip(prot_attrs, priv_classes)], axis=0)
        return df
    
    def format_df_aif360(self, df):
        ds_format = DataSynthesizer(None, None, # No DP/fairness for formatting
                                    self.DP_settings_dict,
                                    self.fair_settings_dict,
                                    self.misc_settings_dict)
        return ds_format.synthesize_DP_fair_df(df)

    def evaluate_dataset_utility(self, df_orig, df_synth):
        out_dict = {}
        for metric in self.dataset_utility_metrics:
            if metric.__name__=="KS_test":
                out_dict.update(metric(df_orig, df_synth))
            else:
                out_dict[metric.__name__] = metric(df_orig, df_synth)
        return out_dict
    
    def evaluate_dataset_fairness(self, df_synth):
        out_dict = {}
        for metric in self.dataset_fairness_metrics:
            if metric.__name__=="mean_outcome_difference":
                df_synth = self.create_comb_prot_attr_columns(df_synth)
                for prot_attr, priv_classes in zip(
                    self.protected_attribute_names_eval, 
                    self.privileged_classes_eval):
                    out_dict[f"{metric.__name__} ({prot_attr})"] = metric(
                        df_synth, self.y_label, self.favorable_classes,
                        prot_attr, priv_classes)
            else: # Default, but likely the above format will end up being used
                out_dict[metric.__name__] = metric(df_synth, self.y_label,
                                                   self.favorable_classes,
                                                   self.protected_attribute_names_eval,
                                                   self.privileged_classes_eval)
        return out_dict
    
    def evaluate_model_utility(self, y_test, y_pred, y_score, X_test):
        out_dict = {}
        X_test = self.create_comb_prot_attr_columns(X_test)
        for metric in self.model_utility_metrics:
            for prot_attr, priv_classes in zip(
                self.protected_attribute_names_eval,
                self.privileged_classes_eval):
                y_test = y_test.set_axis(X_test[prot_attr])
                priv_idx = y_test.index.isin(priv_classes)
                y_hat = y_score if utils.metric_uses_score(metric.__name__) else y_pred

                out_dict[f"{metric.__name__} (overall)"] = metric(y_test, y_hat)
                met_priv = metric(y_test[priv_idx], y_hat[priv_idx])
                out_dict[f"{metric.__name__} (privileged {prot_attr})"] = met_priv
                met_unpriv = metric(y_test[~priv_idx], y_hat[~priv_idx])
                out_dict[f"{metric.__name__} (unprivileged {prot_attr})"] = met_unpriv
                out_dict[f"{metric.__name__} (difference {prot_attr})"] = met_unpriv - met_priv
        return out_dict

    def evaluate_model_fairness(self, y_test, y_pred, X_test):
        out_dict = {}
        X_test = self.create_comb_prot_attr_columns(X_test)
        for metric in self.model_fairness_metrics:
            for prot_attr, priv_classes in zip(
                self.protected_attribute_names_eval,
                self.privileged_classes_eval):
                X_test[prot_attr] = utils.convert_categorical_series_to_binary(
                    X_test[prot_attr], priv_classes)
                y_test = y_test.set_axis(X_test[prot_attr])
                out_dict[f"{metric.__name__} ({prot_attr})"] = metric(
                    y_test, y_pred, prot_attr=prot_attr,
                    priv_group=1, pos_label=1)
        return out_dict

    def simulation_pipeline(self, linear_epsilons_priv,
                            epsilons_fair, n_repeats=30, 
                            results_save_path="simulation_results/",
                            save_incomplete=False):
        SIMULATION_START = time.time()
        # Set up test data
        X_test, y_test = utils.df_to_Xy(self.df_test_aif360_formatted,
                                                    self.y_label)
        y_test = utils.convert_categorical_series_to_binary(
            y_test, self.favorable_classes)
        # Run simulations
        for epsilon_fair in epsilons_fair:
            for linear_epsilon_priv in linear_epsilons_priv:
                epsilon_priv = 10**linear_epsilon_priv if linear_epsilon_priv is not None else None
                n = n_repeats
                if epsilon_fair is None and epsilon_priv is None:
                    n = 1
                    msg = "Simulating original dataset"
                    dataset_str = "Original"
                elif epsilon_fair is None:
                    msg = f"Simulating n={n} repeats using {epsilon_priv:.2}-DP "\
                          f"datasets"
                    dataset_str = "DP"
                elif epsilon_priv is None:
                    msg = f"Simulating n={n} repeats using {epsilon_fair:.2}-fair "\
                          f"datasets"
                    dataset_str = "Fair"
                else:
                    msg = f"Simulating n={n} repeats using {epsilon_priv:.2}-DP, "\
                          f"{epsilon_fair:.2}-fair datasets"
                    dataset_str = "DPFair"
                print("###############################")
                print(msg)
                for _ in tqdm(range(n)):
                    ds = DataSynthesizer(epsilon_priv, epsilon_fair,
                                         self.DP_settings_dict,
                                         self.fair_settings_dict,
                                         self.misc_settings_dict)
                    try:
                        df_train_DPfair = ds.synthesize_DP_fair_df(self.df_train)
                    except Exception as e:
                        print("Iteration failed with exception:", e)
                        continue
                    if df_train_DPfair is None: # synthesis failed
                        continue

                    # Record settings used
                    single_sim_dict = dict()
                    single_sim_dict["Linear epsilon (privacy)"] = linear_epsilon_priv
                    single_sim_dict["Epsilon (privacy)"] = epsilon_priv
                    single_sim_dict["Epsilon (fairness)"] = epsilon_fair
                    single_sim_dict["Dataset"] = dataset_str

                    # Record dataset metrics
                    single_sim_dict.update(self.evaluate_dataset_utility(
                        self.df_train_aif360_formatted.copy(),
                        df_train_DPfair.copy()))
                    single_sim_dict.update(self.evaluate_dataset_fairness(
                        df_train_DPfair.copy()
                    ))

                    # Train and evaluate models
                    X_train_DPfair, y_train_DPfair = utils.df_to_Xy(
                        df_train_DPfair, self.y_label)
                    y_train_DPfair = utils.convert_categorical_series_to_binary(
                        y_train_DPfair, self.favorable_classes)
                    for model in self.models:
                        single_sim_dict["Model"] = type(model).__name__
                        model.fit(X_train_DPfair, y_train_DPfair)
                        y_pred = model.predict(X_test)
                        y_score = model.predict_proba(X_test)[:,1]

                        # Record model metrics
                        single_sim_dict.update(self.evaluate_model_utility(
                            y_test.copy(), y_pred, y_score, X_test.copy()
                        ))
                        single_sim_dict.update(self.evaluate_model_fairness(
                            y_test.copy(), y_pred, X_test.copy()
                        ))

                    # Update results
                    self.update_results_dict(single_sim_dict)

                # Print time passed
                h, m, s = utils.get_time_passed(SIMULATION_START)
                print(f"Time passed: {h} hours, {m} minutes, and {s} seconds\n")

                # Save incomplete results
                if save_incomplete:
                    self.save_results(results_save_path + self.dataset + "/incomplete/")

        # Save complete results
        self.save_results(results_save_path + self.dataset + "/complete/")