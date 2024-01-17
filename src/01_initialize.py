import os
import argparse
from utils.management import resolve_experiment, create_experiment, create_run, load_config_to_run
from utils.logging import print_

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_name", help="Name of the run to initialize", default=None)
    parser.add_argument("-e", "--experiment_name", help="Name of the experiment", default=None)
    parser.add_argument("-c", "--config", help="Configuration to load for the run", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    #clear_cmd()
    args = arguments()
    experiment_name, run_name = resolve_experiment(exp_name=args.experiment_name, run_name=args.run_name)
    if experiment_name is None:
        print_("Cannot initialize experiment without name.", "error")
        exit()
    if run_name is None:
        create_experiment(experiment_name)
    else:
        create_run(experiment_name, run_name)
        if args.config is not None:
            load_config_to_run(args.config, args.experiment_name, args.run_name)


