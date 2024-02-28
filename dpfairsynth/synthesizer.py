from itertools import combinations
from scipy import sparse
import json

from mbi import FactoredInference
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools

# Custom files
import formatters
from dp_mechanisms import Laplace_mech
import adult_preprocessing

class DataSynthesizer():
    def __init__(self, epsilon_DP=None, epsilon_fair=None, 
                 custom_aif360_conversion='adult'):
        self.epsilon_DP = epsilon_DP
        self.epsilon_fair = epsilon_fair

        if custom_aif360_conversion=='adult':
            self.aif360_conversion = adult_preprocessing.custom_aif360_conversion
        elif callable(custom_aif360_conversion):
            self.aif360_conversion = custom_aif360_conversion
        else:
            self.aif360_conversion = None

    def set_DP_settings(self, DP_settings_dict):
        for k, v in DP_settings_dict.items():
            setattr(self, k, v)
            
        if DP_settings_dict["domain_dict"]=='adult':
            with open("preprocessed_data/adult/adult_domain.json") as f:
                self.domain_dict = json.load(f)

    def set_fair_settings(self, fair_settings_dict):
        for k, v in fair_settings_dict.items():
            setattr(self, k, v)

        if fair_settings_dict["custom_distortion"]=='adult':
            self.custom_distortion = adult_preprocessing.custom_distortion

    def set_required_cols(self, df):
        if self.aif360_conversion:
            df = self.aif360_conversion(df)
        
        # Convert to AIF360 StandardDataset and back to dataframe
        std_dataset = StandardDataset(df.copy(), label_name=self.y_label, 
                favorable_classes=self.favorable_classes,
                protected_attribute_names=self.protected_attribute_names,
                privileged_classes=self.privileged_classes,
                categorical_features=self.categorical_features,
                features_to_keep=self.features_to_keep, 
                metadata=self.metadata)
        df = std_dataset.convert_to_dataframe()[0]
        self.required_cols = df.columns

    def df_to_standard_dataset(self, df):
        return StandardDataset(df, label_name=self.y_label, 
                favorable_classes=self.favorable_classes,
                protected_attribute_names=self.protected_attribute_names,
                privileged_classes=self.privileged_classes,
                categorical_features=self.categorical_features,
                features_to_keep=self.features_to_keep, 
                metadata=self.metadata)
    
    def correct_missing_cols(self, df, missing_cols):
        df[list(missing_cols)] = 0.0
        return df[self.required_cols]

    def synthesize_DP_fair_df(self, df, 
                              DP_settings_dict={
                                  "domain_dict": 'adult',
                                  "cliques": 'all 2-way',
                              },
                              fair_settings_dict={
                                  "y_label": "income>50K",
                                  "favorable_classes": [1],
                                  "protected_attribute_names": ['race'],
                                  "privileged_classes": [[1]],
                                  "categorical_features": [
                                      'age-decade',
                                      'education-reduced'
                                  ],
                                  "features_to_keep": [
                                      'age-decade',
                                      'education-reduced',
                                      'race-reduced',
                                      'sex-num',
                                      'income>50K',
                                  ],
                                  "metadata": {'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
                                               'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'}]},
                                  "custom_distortion": 'adult',
                                  "verbose": False,
                              }):
        # Set DP/fairness settings
        self.set_DP_settings(DP_settings_dict)
        self.set_fair_settings(fair_settings_dict)

        # Define columns necessary for synthetic data
        self.set_required_cols(df)

        # DP data synthesis
        if self.epsilon_DP:
            df = self.synthesize_DP_df(df)

        # Convert df to format needed for AIF360 package
        if self.aif360_conversion:
            df = self.aif360_conversion(df)
        
        # Convert to AIF360 StandardDataset
        std_dataset = self.df_to_standard_dataset(df)

        # Ensure no missing columns
        tmp_df = std_dataset.convert_to_dataframe()[0]
        missing_cols = set(self.required_cols) - set(tmp_df.columns)
        if missing_cols:
            tmp_df = self.correct_missing_cols(tmp_df, missing_cols)
            std_dataset = self.df_to_standard_dataset(tmp_df)

        # Fairness transformation
        if self.epsilon_fair:
            std_dataset = self.synthesize_fair_df(std_dataset)
        
        # Account for random failure to converge, etc.
        if std_dataset is None:
            print("Fairness transformation failed.")
            return None
        
        # Convert back to df
        df = std_dataset.convert_to_dataframe()[0]

        # Ensure no missing columns again
        missing_cols = set(self.required_cols) - set(df.columns)
        if missing_cols:
            df = self.correct_missing_cols(df, missing_cols)

        return df

    def synthesize_DP_df(self, df):
        # Convert to MBI Dataset format
        mbi_dataset = formatters.df_to_MBIDataset(df, self.domain_dict)

        # cliques can also be given as an explicit list of n-way tuples
        cliques = self.cliques
        if cliques=='all 2-way': 
            cliques = list(combinations(mbi_dataset.df.columns, 2))
        
        # Divide total epsilon over all measurements
        epsilon_split = self.epsilon_DP / (len(mbi_dataset.domain) + len(cliques))
        # Sensitivity of projection measurements (occurrence frequency count vectors)
        sens = 2.0 

        # Measure one-dimensional marginals
        measurements = []
        for col in mbi_dataset.domain:
            x = mbi_dataset.project(col).datavector()
            y = Laplace_mech(x, epsilon_split, sens)
            I = sparse.eye(x.size)
            measurements.append( (I, y, sens/epsilon_split, (col,)) )
            
        # Measure multi-dimensional marginals (cliques)
        for cl in cliques:
            x = mbi_dataset.project(cl).datavector()
            y = Laplace_mech(x, epsilon_split, sens)
            I = sparse.eye(x.size)
            measurements.append( (I, y, sens/epsilon_split, cl) )

        # GENERATE synthetic data using Private-PGM 
        engine = FactoredInference(mbi_dataset.domain, iters=2500)
        model = engine.estimate(measurements, total=mbi_dataset.records)
        DP_mbi_dataset = model.synthetic_data()

        # Convert to df
        df_DP = formatters.MBIDataset_to_df(DP_mbi_dataset)

        return df_DP

    def synthesize_fair_df(self, std_dataset):
        optim_options = {
            "distortion_fun": self.custom_distortion,
            "epsilon": self.epsilon_fair,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }

        OP = OptimPreproc(OptTools, optim_options, verbose=self.verbose)

        try:
            OP = OP.fit(std_dataset)
        except:
            print("Fairness transformation failed to converge...")
            return None

        std_dataset_fair = OP.transform(std_dataset, transform_Y=True)
        try:
            std_dataset_fair = std_dataset.align_datasets(std_dataset_fair)
        except:
            print("Fairness-transformed data failed to align with original...")
            return None
        return std_dataset_fair