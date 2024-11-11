# DP synthetic data stuff
from mbi import Dataset, Domain

def df_to_MBIDataset(df, domain_dict):
    return Dataset(df, Domain.fromdict(domain_dict))

def MBIDataset_to_df(mbi_dataset):
    return mbi_dataset.df