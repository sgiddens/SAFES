import time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from aif360.sklearn.metrics import statistical_parity_difference, average_odds_difference

from custom_metrics import mean_outcome_difference, KS_test
from synthesizer import DataSynthesizer
import adult_synthesizing
import adult_preprocessing
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
                 ],
                 model_fairness_metrics=[
                     statistical_parity_difference,
                     average_odds_difference,
                 ],
                 random_state=50,
                 ):
        # Load and store train/test dataframes
        if dataset=="adult":
            df, domain_dict = adult_preprocessing.load_preprocessed_adult_data()
            # Define settings
            (self.DP_settings_dict, 
            self.fair_settings_dict, 
            self.misc_settings_dict) = adult_synthesizing.define_settings()
            self.DP_settings_dict["domain_dict"] = domain_dict
        else:
            raise ValueError("dataset value not currently supported.")
        self.y_label = self.fair_settings_dict["y_label"]
        self.favorable_classes = self.fair_settings_dict["favorable_classes"]
        self.protected_attribute_names = self.fair_settings_dict["protected_attribute_names"]
        self.privileged_classes = self.fair_settings_dict["privileged_classes"]

        X, y = utils.df_to_Xy(df, self.y_label)
        (X_train, X_test, 
         y_train, y_test) = train_test_split(X, y, random_state=random_state)
        self.df_train = utils.Xy_to_df(X_train, y_train)
        self.df_test = utils.Xy_to_df(X_test, y_test)

        self.models = models

        self.dataset_fairness_metrics = dataset_fairness_metrics
        self.dataset_utility_metrics = dataset_utility_metrics
        self.model_fairness_metrics = model_fairness_metrics
        self.model_utility_metrics = model_utility_metrics

        # Initialize results dictionary
        self.reset_results_dict()

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
                for prot_attr in self.protected_attribute_names:
                    results_dict.update({f"{metric.__name__} "\
                                         f"({prot_attr})": []})
                if len(self.protected_attribute_names)>1:
                    results_dict.update({f"{metric.__name__} "\
                                         f"({'*'.join(self.protected_attribute_names)})": []})
            else:
                results_dict.update({f"{metric.__name__}": []})

        results_dict.update({f"{metric.__name__} (overall)": []
                            for metric in self.model_utility_metrics})
        results_dict.update({f"{metric.__name__} (privileged)": []
                            for metric in self.model_utility_metrics})
        results_dict.update({f"{metric.__name__} (unprivileged)": []
                            for metric in self.model_utility_metrics})
        results_dict.update({f"{metric.__name__} (difference)": []
                            for metric in self.model_utility_metrics})
        results_dict.update({f"{metric.__name__}": []
                             for metric in self.model_fairness_metrics})
        self.results_dict = results_dict

    def update_results_dict(self, new_results):
        if set(self.results_dict.keys()!=set(new_results.keys())):
            print("Keys in new results dictionary don't match full results "\
                  "dictionary. Full results dictionary was not updated.")
            return
        for k in self.results_dict.keys():
            self.results_dict[k].append(new_results[k])

    def save_results(self):
        pass

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
                protected_attribute_names = list(self.protected_attribute_names)
                privileged_classes = list(self.privileged_classes)
                if len(self.protected_attribute_names)>0:
                    comb_prot_attr = '*'.join(self.protected_attribute_names)
                    df_synth[comb_prot_attr] = np.prod([np.where(
                        df_synth[attr].isin(priv_class), 1, 0) 
                        for attr, priv_class in zip(self.protected_attribute_names,
                                                    self.privileged_classes)
                    ], axis=0)
                    protected_attribute_names.append(comb_prot_attr)
                    privileged_classes.append([1])
                for prot_attr, priv_classes in zip(
                    protected_attribute_names, privileged_classes):
                    out_dict[f"{metric.__name__} ({prot_attr})"] = metric(
                        df_synth, self.y_label, self.favorable_classes,
                        prot_attr, priv_classes)
            else:
                out_dict[metric.__name__] = metric(df_synth, self.y_label,
                                                   self.favorable_classes,
                                                   self.protected_attribute_names,
                                                   self.privileged_classes)
        return out_dict
    
    def evaluate_model_utility(self):
        pass

    def evaluate_model_fairness(self):
        pass

    def simulation_pipeline(self, linear_epsilons_priv,
                            epsilons_fair, n_repeats=30, 
                            results_path="simulation_results/"):
        SIMULATION_START = time.time()
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
                    msg = f"Simulating n={n} repeats using {epsilon_fair:.2}-DP "\
                          f"datasets"
                    dataset_str = "Fair"
                else:
                    msg = f"Simulating n={n} repeats using {epsilon_priv:.2}-DP, "\
                          f"{epsilon_fair:.2}-fair datasets"
                    dataset_str = "DPFair"
                print(msg)
                for _ in tqdm(range(n)):
                    ds = DataSynthesizer(epsilon_priv, epsilon_fair,
                                         self.DP_settings_dict,
                                         self.fair_settings_dict,
                                         self.misc_settings_dict)
                    df_train_DPfair = ds.synthesize_DP_fair_df(self.df_train)
                    if df_train_DPfair is None:
                        continue

                    # Record settings used
                    single_sim_dict = dict()
                    single_sim_dict["Linear epsilon (privacy)"] = linear_epsilon_priv
                    single_sim_dict["Epsilon (privacy)"] = epsilon_priv
                    single_sim_dict["Epsilon (fairness)"] = epsilon_fair
                    single_sim_dict["Dataset"] = dataset_str
                    # Record dataset metrics
                    single_sim_dict.update(self.evaluate_dataset_utility(
                        self.df_train.copy(),
                        df_train_DPfair.copy()))
                    single_sim_dict.update(self.evaluate_dataset_fairness(
                        df_train_DPfair.copy()
                    ))

                    # Train and evaluate models
                    X_train_DPfair, y_train_DPfair = utils.df_to_Xy(
                        df_train_DPfair, self.y_label)
                    for model in self.models:
                        single_sim_dict["Model"] = type(model).__name__
                        model.fit(X_train_DPfair, y_train_DPfair)
                        y_pred = model.predict(X_test)
                    # Record model metrics
                    self.evaluate_model_utility()
                    self.evaluate_model_fairness()

                    



                    self.update_results_dict(single_sim_dict)