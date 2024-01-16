import os
import argparse
import torch

from utils.management import resolve_experiment
from utils.logging import print_
from training import Trainer

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_name", help="Name of the run to initialize", default=None)
    parser.add_argument("-e", "--experiment_name", help="Name of the experiment", default=None)
    parser.add_argument("--log", action='store_true', default=False, help="Log the training process")
    return parser.parse_args()


if __name__ == "__main__":
    #clear_cmd()
    args = arguments()
    experiment_name, run_name = resolve_experiment(exp_name=args.experiment_name, run_name=args.run_name)
    if experiment_name is None or run_name is None:
        print_("Experiment or run name not provided. Cannot start training!", "error")
        exit("-1")
    
    device ="cuda" if torch.cuda.is_available() else "cpu"
    print_(f"Device: {device}")
    # Initialize the trainer
    trainer = Trainer(experiment_name, run_name, device)
    # Initialize the logger
    trainer.initialize_logging()
    # Initialize the model
    trainer.initialize_model()
    # Initialize the dataset
    trainer.initialize_data()
    # Initialize the optimizer
    trainer.initialize_optimization()
    # Train the model
    trainer.train()

