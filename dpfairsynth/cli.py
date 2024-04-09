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

    parser.add_argument(
        "--draw-graph",
        action="store_true",
        help="Save figure representation of produced graphical model from DP data synthesizer."
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
            print(df_synth.columns)
        else:
            print("Chosen dataset not supported for synthesizing.")
    elif args.draw_graph:
        if args.epsilon_DP is None:
            print("epsilon_DP must be provided to draw graph.")
            return
        if args.dataset=='adult':
            adult_synthesizing.draw_graph_pipeline(args.epsilon_DP)
        elif args.dataset=='compas':
            compas_synthesizing.draw_graph_pipeline(args.epsilon_DP)
        else:
            print("Chosen dataset not supported for graph drawing.")
    elif args.run_simulations:
        # warm_start_file = "simulation_results/compas/incomplete/simulations_2024-04-01_23-21-23.csv"
        warm_start_file = None
        linear_epsilons_priv = [None] + list(np.linspace(-2, 1, 7))

        epsilons_fair_adult = [None, 0.025, 0.05] # Default for adult
        epsilons_fair_compas = [None, 0.08, 0.12] # Default for compas
        if args.dataset=='adult':
            epsilons_fair = epsilons_fair_adult
        elif args.dataset=='compas':
            epsilons_fair = epsilons_fair_compas
        # epsilons_fair = [] # Custom if desired
        
        dpfair_eval = DPFairEvaluator(args.dataset, 
                                      warm_start_file=warm_start_file)
        dpfair_eval.simulation_pipeline(linear_epsilons_priv,
                                        epsilons_fair, n_repeats=35,
                                        save_incomplete=False)
        print("Done!")
    else:
        print("No valid command line argument present.")