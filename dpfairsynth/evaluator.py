import time
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
    def __init__(self, dataset, prot_attr, privileged_classes, 
                 priv_group, pos_label, y_label,
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
        X, y = utils.df_to_Xy(df, y_label)
        (X_train, X_test, 
         y_train, y_test) = train_test_split(X, y, random_state=random_state)
        self.df_train = utils.Xy_to_df(X_train, y_train)
        self.df_test = utils.Xy_to_df(X_test, y_test)

        self.models = models

        self.prot_attr = prot_attr
        self.privileged_classes = privileged_classes
        self.priv_group = priv_group
        self.pos_label = pos_label
        self.y_label = y_label

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
                results_dict.update({f"{metric.__name__}: []"})
        results_dict.update({f"{metric.__name__}": [] 
                             for metric in self.dataset_fairness_metrics})
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
        pass

    def save_results(self):
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
                elif epsilon_fair is None:
                    msg = f"Simulating n={n} repeats using {epsilon_priv:.2}-DP "\
                          f"datasets"
                elif epsilon_priv is None:
                    msg = f"Simulating n={n} repeats using {epsilon_fair:.2}-DP "\
                          f"datasets"
                else:
                    msg = f"Simulating n={n} repeats using {epsilon_priv:.2}-DP, "\
                          f"{epsilon_fair:.2}-fair datasets"
                print(msg)
                for _ in tqdm(range(n)):
                    data_synthesizer = DataSynthesizer(epsilon_priv, epsilon_fair)


                    # Record settings used
                    self.results_dict["Linear epsilon (privacy)"].append(linear_epsilon_priv)
                    self.results_dict["Epsilon (privacy)"].append(epsilon_priv)
                    self.results_dict["Epsilon (fairness)"].append(epsilon_fair)
                    dataset_str = ""
                    if epsilon_priv is not None:
                        dataset_str+="DP"
                    if epsilon_fair is not None:
                        dataset_str+="Fair"
                    dataset_str = "Original" if dataset_str=="" else dataset_str
                    self.results_dict["Dataset"].append(dataset_str)
