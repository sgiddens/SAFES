import numpy as np 
import pandas as pd
from itertools import combinations
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score as original_auc
from sklearn.metrics import recall_score

def mean_outcome_difference(df, y_label, favorable_classes,
                            prot_attr, privileged_classes):
    y = df[y_label]
    A = df[prot_attr]
    
    unprivileged_group_mean = (y[~A.isin(privileged_classes)].isin(
        favorable_classes)).mean()
    privileged_group_mean = (y[A.isin(privileged_classes)].isin(
        favorable_classes)).mean()
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

def roc_auc_score(y_true, y_score):
    """Custom roc_auc_score to handle errors if all y_true 
    are one class by random chance during simulations."""
    try:
        val = original_auc(y_true, y_score)
    except:
        val = np.nan
    return val

def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def TVD(df_real, df_synth, cols):
    return (df_real[cols].value_counts(normalize=True) - 
            df_synth[cols].value_counts(normalize=True)).abs().sum()/2

def n_way_TVDs(df_real, df_synth, n):
    combs = list(combinations(df_real.columns, n))
    out_dict = {}
    for c in combs:
        out_dict[f"TVD_{n}way_{'+'.join(list(c))}"] = TVD(df_real, df_synth, list(c))
    return out_dict

def one_way_TVDs(df_real, df_synth):
    return n_way_TVDs(df_real, df_synth, n=1)

def two_way_TVDs(df_real, df_synth):
    return n_way_TVDs(df_real, df_synth, n=2)

def three_way_TVDs(df_real, df_synth):
    return n_way_TVDs(df_real, df_synth, n=3)