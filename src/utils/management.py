"""
    Functions that handle the complete communication with the local experiment directory.
    The functionalities include:
     - Loading and saving configs
     - Loading and saving model checkpoints
     - Initialising experiments
     - Initialising and deleting runs
     - Clear run data

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import os
import shutil

import torch

import json

from .logging import print_

from typing import Dict, List, Tuple, Optional

###--- Experiment Management Scripts ---###
# These functions handle all interaction from outside with the experiments
# We can simply initialize experiments, load and save configs, and load model checkpoints

###--- Experiments Directory ---###

def create_experiment(experiment_name):
    """ Create a new experiment in the experiments directory. """
    path = os.path.join(os.getcwd(), 'experiments', experiment_name)
    if os.path.exists(path):
        print_('Experiment directory already exists')
        return -1
    os.makedirs(path)
    print_(f'Experiment directory created at: {path}')
    return 1

def create_run(experiment_name, run_name):
    """ Create a new run in an existing experiment directory. """
    path = os.path.join(os.getcwd(), 'experiments', experiment_name, run_name)
    if os.path.exists(path):
        print_('Run directory already exists')
        return -1
    os.makedirs(path)
    os.makedirs(os.path.join(path, 'logs'))
    os.makedirs(os.path.join(path, 'checkpoints'))
    os.makedirs(os.path.join(path, 'visualizations'))
    print_(f'Run directory created at: {path}')
    return 1

def clear_run(experiment_name, run_name):
    """ Remove a run directory. """
    path = os.path.join(os.getcwd(), 'experiments', experiment_name, run_name)
    if not os.path.exists(path):
        print_('Run directory does not exist')
        return -1
    shutil.rmtree(path)
    os.makedirs(path)
    print_(f'Run directory removed at: {path}')
    return 1

def clear_logs(experiment_name, run_name):
    """ Clear the local log files of a run. """
    path = os.path.join(os.getcwd(), 'experiments', experiment_name, run_name, 'logs')
    if not os.path.exists(path):
        print_('Logs directory does not exist')
        return -1
    shutil.rmtree(path)
    os.makedirs(path)
    print_(f'Logs directory removed at: {path}')
    return 1

def get_env_setup():
    """ Get experiment and run name from environment."""
    exp_name = None
    run_name = None
    if 'CURRENT_EXP' in os.environ:
        exp_name = os.environ['CURRENT_EXP']
    if 'CURRENT_RUN' in os.environ:
        run_name = os.environ['CURRENT_RUN']
    return exp_name, run_name
    
def resolve_experiment(exp_name: Optional[str]=None, run_name: Optional[str]=None):
    """ 
        Resolve provided and environment experiment and run names.
        Provided experiment and run names always override environment experiment and run names.
    """

    env_exp_name, env_run_name = get_env_setup()
    if exp_name is None:
        exp_name = env_exp_name
    if run_name is None:
        run_name = env_run_name
    if exp_name is None or run_name is None:
        print_('Experiment or run name is not set', 'warn')
    return exp_name, run_name

###--- Configurations ---###

def load_config_to_run(config_name, exp_name, run_name):
    """ Load a configuration from the /configurations directory to a run."""
    config = load_config(config_name)
    save_config(config, exp_name, run_name)

def load_config_from_run(exp_name, run_name):
    """ Load the configuration from a run directory. """
    path = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'config.json')
    if not os.path.exists(path):
        print_('Config file does not exist')
        return None
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def load_config(config_name):
    """ Load a configuration from the /configurations directory. """
    with open(os.path.join(os.getcwd(), 'configurations', config_name+".json"), 'r') as f:
        config = json.load(f)
    return config

def save_config(config, exp_name, run_name):
    """ Save a configuration to a run directory. """
    with open(os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


###--- Checkpoint Loading ---###

def load_model_from_checkpoint(exp_name,
                                run_name,
                                 model,
                                  epoch, 
                                   optimizer=None,
                                    scheduler=None,
                                     device='cpu'):
    cp_dir = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
    try:
        cp_data = torch.load(cp_dir, map_location=device)
        model.load_state_dict(cp_data['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(cp_data['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(cp_data['scheduler_state_dict'])
    except Exception as e:
        print_(f'Checkpoint could not been load from: {cp_dir}', 'error')
        print_(f'Exception: {e}', 'error')
        return
    print_(f'Model checkpoint was load from: {cp_dir}')

def load_model_from_path(path: str, model: torch.nn.Module, optimizer=None, scheduler=None, device: str = 'cpu'):
    try:
        data = torch.load(path, map_location=torch.device(device))
        model.load_state_dict(data['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(data['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(data['scheduler_state_dict'])
    except Exception as e:
        print_(f'Checkpoint could not been load from: {path}', 'error')
        print_(f'Exception: {e}', 'error')
        return
    print_(f'Model checkpoint was load from: {path}')
    

###--- Data ---###

def reset_data():
    if os.path.exists(os.path.join(os.getcwd(), 'data')):
        shutil.rmtree(os.path.join(os.getcwd(), 'data'))
    os.makedirs(os.path.join(os.getcwd(), 'data'))
    print_('Data directory resetted')
    return 1