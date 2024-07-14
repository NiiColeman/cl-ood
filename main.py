# main.py

import argparse
import yaml
from experiments.train import main as train_main
from experiments.evaluate import main as evaluate_main
from tqdm import tqdm


def run_experiment(config_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    print("Starting auxiliary training...")
    train_main(config)
    print("End of Training")

    # print("\nStarting fine-tuning and evaluation...")
    # evaluate_main(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OOD Generalization Experiment")
    parser.add_argument("--config", type=str, default="configs/custom_config.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    run_experiment(args.config)