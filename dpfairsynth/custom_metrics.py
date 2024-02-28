import numpy as np 
import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp

# TODO: Streamline this function
def mean_outcome_difference(df, y_label, favorable_classes,
                            prot_attr, privileged_classes):
    favorable_class = 1
    unfavorable_class = 0
    privileged_class = 1
    unprivileged_class = 0

    # Define y as binary
    y = df[y_label]
    tmp_y = y.copy()
    tmp_y = unfavorable_class
    for c in favorable_classes:
        tmp_y[y==c] = favorable_class
    y = tmp_y

    # Define privileged attribute(s) as binary
    if isinstance(prot_attr, list):
        df["*".join(prot_attr)] = np.ones_like(df[prot_attr[0]])
        for attr, priv_class in zip(prot_attr, privileged_classes):
            A = df[attr]
            tmp_A = A.copy()
            tmp_A = unprivileged_class
            for p in priv_class:
                tmp_A[A==p] = privileged_class
            A = tmp_A
            df["*".join(prot_attr)] *= A
        prot_attr = "*".join(prot_attr)
        A = df[prot_attr]
    else:
        A = df[prot_attr]
        tmp_A = A.copy()
        tmp_A = unprivileged_class
        for p in privileged_classes:
            tmp_A[A==p] = privileged_class
        A = tmp_A
    
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