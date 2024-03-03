import argparse
import logging
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

import adult_preprocessing
import adult_synthesizing


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
        choices=['adult'],
        default='adult',
        help="The dataset to be used for preprocessing or "\
             "generating synthetic data. Defaults to 'adult'.", 
    )

    parser.add_argument(
        '--synthesize-data',
        action="store_true",
        help="Synthesize a DPFair dataset as indicated by the "\
             "--dataset, --epsilon_DP, and --epsilon_fair flags."
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
        else:
            print("Chosen dataset not supported for preprocessing.")
    elif args.synthesize_data:
        if args.dataset=='adult':
            df_synth = adult_synthesizing.synthesizing_pipeline(args.epsilon_DP,
                                                                args.epsilon_fair)
            print(df_synth.head())
            # df_synth.to_csv("simulation_results/DPfair.csv", index=False)
        else:
            print("Chosen dataset not supported for synthesizing.")
    else:
        print("No valid command line argument present.")