import argparse
import logging
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import adult.adult_preprocessing as adult_preprocessing
import adult.adult_synthesizing as adult_synthesizing
import compas.compas_preprocessing as compas_preprocessing
import compas.compas_synthesizing as compas_synthesizing
from evaluator import DPFairEvaluator


def create_parser():
    """Create and configure the argparse parser."""
    parser = argparse.ArgumentParser(description="Argparser")

    parser.add_argument(
        "--preprocess-data", 
        action="store_true",
        help="Run the preprocessor for the dataset indicated by --dataset flag.",
    )

    parser.add_argument(
        "--dataset",
        action="store",
        type=str,
        choices=['adult', 'compas'],
        default='adult',
        help="The dataset to be used for preprocessing or "\
             "generating synthetic data. Defaults to 'adult'.", 
    )

    parser.add_argument(
        "--synthesize-data",
        action="store_true",
        help="Synthesize a DPFair dataset as indicated by the "\
             "--dataset, --epsilon_DP, and --epsilon_fair flags."
    )

    parser.add_argument(
        "--run-simulations",
        action="store_true",
        help="Run simulations to evaluate the DPFair data synthesis approach."
    )

    parser.add_argument(
        "--epsilon-DP",
        action="store",
        type=float,
        help="Privacy budget for synthesizing data with "\
             "epsilon-DP guarantees."
    )

    parser.add_argument(
        "--epsilon-fair",
        action="store",
        type=float,
        help="Fairness parameter for transforming data "\
             "according to epsilon-fair constraint."
    )

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.preprocess_data:
        if args.dataset=='adult':
            adult_preprocessing.preprocessing_pipeline()
        elif args.dataset=='compas':
            compas_preprocessing.preprocessing_pipeline()
        else:
            print("Chosen dataset not supported for preprocessing.")
    elif args.synthesize_data:
        if args.dataset=='adult':
            df_synth = adult_synthesizing.synthesizing_pipeline(args.epsilon_DP,
                                                                args.epsilon_fair)
            print(df_synth.head())
            # df_synth.to_csv("synthesized_datasets/DPfair.csv", index=False)
        elif args.dataset=='compas':
            df_synth = compas_synthesizing.synthesizing_pipeline(args.epsilon_DP,
                                                                 args.epsilon_fair)
            print(df_synth)
        else:
            print("Chosen dataset not supported for synthesizing.")
    elif args.run_simulations:
        # warm_start_file = "simulation_results/adult/incomplete/simulations_2024-03-04_17-49-31.csv"
        warm_start_file = None
        linear_epsilons_priv = [None] + list(np.linspace(-2, 1, 7))
        epsilons_fair = [None, 0.08, 0.12]
        dpfair_eval = DPFairEvaluator(args.dataset, 
                                      warm_start_file=warm_start_file)
        dpfair_eval.simulation_pipeline(linear_epsilons_priv,
                                        epsilons_fair, n_repeats=35,
                                        save_incomplete=True)
        print("Done!")
    else:
        print("No valid command line argument present.")