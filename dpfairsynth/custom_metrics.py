import numpy as np 
import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp

def mean_outcome_difference(df, y_label, favorable_classes,
                            prot_attr, privileged_classes):
    favorable_class = 1
    unfavorable_class = 0
    privileged_class = 1
    unprivileged_class = 0

    # Define y as binary
    y = df[y_label]
    y.replace(favorable_classes, favorable_class, inplace=True)
    y[y!=favorable_class] = unfavorable_class

    # Define privileged attribute as binary
    A = df[prot_attr]
    A.replace(privileged_classes, privileged_class, inplace=True)
    A[A!=privileged_class] = unprivileged_class
    
    unprivileged_group_mean = (y[A!=privileged_class]==favorable_class).mean()
    privileged_group_mean = (y[A==privileged_class]==favorable_class).mean()
    return unprivileged_group_mean - privileged_group_mean

def KS_test(df_real, df_synth):
    # Add labels and combine original and synthetic data
    df_real["Synthetic"] = 0
    df_synth["Synthetic"] = 1
    df_combined = pd.concat([df_real, df_synth])
    X, y = df_combined.drop("Synthetic", axis=1), df_combined["Synthetic"]
    X, y = shuffle(X, y)#, random_state=42)
    
    # Calculate propensity scores
    model = LogisticRegression()
    model.fit(X, y)
    propensity_scores = model.predict_proba(X)[:, 1]
    
    # Separate propensity scores
    propensity_scores_real = propensity_scores[y==0]
    propensity_scores_synth = propensity_scores[y==1]
    
    # Compute KS distance between two empirical CDFs
    ks_statistic, ks_pval = ks_2samp(propensity_scores_real, propensity_scores_synth)
    out_dict = {
        "KS Statistic": ks_statistic,
        "KS p-value": ks_pval,
    }
    return out_dict