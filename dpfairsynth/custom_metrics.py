# import numpy as np 
import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp

def mean_outcome_difference(df, prot_attr, y_label,
                            priv_group, pos_label):
    # if isinstance(prot_attr, list):
    #     df["*".join(prot_attr)] = np.ones_like(df[prot_attr[0]])
    #     for attr in prot_attr:
    #         df["*".join(prot_attr)] *= df[attr]
    #     prot_attr = "*".join(prot_attr)
    y = df[y_label]
    unprivileged_group_mean = (y[df[prot_attr]!=priv_group]==pos_label).mean()
    privileged_group_mean = (y[df[prot_attr]==priv_group]==pos_label).mean()
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