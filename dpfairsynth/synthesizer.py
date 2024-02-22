from itertools import combinations
from scipy import sparse

from mbi import FactoredInference

# Custom files
import formatters
from dp_mechanisms import Laplace_mech

class DataSynthesizer():
    def __init__(self, epsilon_DP=None, epsilon_fair=None):
        self.epsilon_DP = epsilon_DP
        self.epsilon_fair = epsilon_fair

    def synthesize_DP_fair_df(self, df, domain_dict, cliques='all 2-way'):
        if self.epsilon_DP is not None:
            df = self.synthesize_DP_df(df, domain_dict, cliques)

    def synthesize_DP_df(self, df, domain_dict, cliques='all 2-way'):
        # Convert to MBI Dataset format
        mbi_dataset = formatters.df_to_MBIDataset(df, domain_dict)

        # cliques can also be given as an explicit list of n-way tuples
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
