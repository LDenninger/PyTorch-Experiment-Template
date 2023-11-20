"""
    This module capsules the logging and complete communication with the WandB API.
    It can be used for either logging or load previously logged data.

    Some parts of the logging module were adapted from: https://github.com/angelvillar96/TemplaTorch

    Author: Luis Denninger <l_denninger@uni-bonn.de>

"""

import numpy as np
import json
import torch
import torch.nn as nn
import torchvision
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union, Literal
import traceback
from datetime import datetime
import os
import git
from pathlib import Path as P
import csv


#####===== Logging Decorators =====#####

def log_function(func):
    """
        Decorator to catch a function in case of an exception and writing the output to a log file.
    """
    def try_call_log(*args, **kwargs):
        """
            Calling the function but calling the logger in case an exception is raised.
        """
        try:
            if(LOGGER is not None):
                message = f"Calling: {func.__name__}..."
                LOGGER.log_info(message=message, message_type="info")
            return func(*args, **kwargs)
        except Exception as e:
            if(LOGGER is None):
                raise e
            message = traceback.format_exc()
            print_(message, message_type="error")
            exit()
    return try_call_log

def for_all_methods(decorator):
    """
        Decorator that applies a decorator to all methods inside a class.
    """
    def decorate(cls):
        for attr in cls.__dict__:  # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

def emergency_save(f):
    """
    Decorator for saving a model in case of exception, either from code or triggered.
    Use for decorating the training loop:
        @setup_model.emergency_save
        def train_loop(self):
    """

    def try_call_except(*args, **kwargs):
        """ Wrapping function and saving checkpoint in case of exception """
        try:
            return f(*args, **kwargs)
        except (Exception, KeyboardInterrupt):
            print_("There has been an exception. Saving emergency checkpoint...")
            self_ = args[0]
            if hasattr(self_, "model") and hasattr(self_, "optimizer"):
                fname = f"emergency_checkpoint_epoch_{self_.epoch}.pth"
                save_checkpoint(
                    model=self_.model,
                    optimizer=self_.optimizer,
                    scheduler=self_.scheduler,
                    epoch=self_.epoch,
                    exp_path=self_.exp_path,
                    savedir="models",
                    savename=fname
                )
                print_(f"  --> Saved emergency checkpoint {fname}")
            message = traceback.format_exc()
            print_(message, message_type="error")
            exit()

    return try_call_except

def print_(message, message_type="info", file_name: str=None, path_type: Literal['log', 'plot', 'checkpoint', 'visualization'] = None):
    """
    Overloads the print method so that the message is written both in logs file and console
    """
    print(message)
    if(LOGGER is not None):
        if file_name is None:
            LOGGER.log_info(message, message_type)
        elif file_name is not None and path_type is not None:
            LOGGER.log_to_file(message, file_name, path_type)
    return


def log_info(message, message_type="info"):
    if(LOGGER is not None):
        LOGGER.log_info(message, message_type)
    return

def get_current_git_hash():
    """ Obtaining the hexadecimal last commited git hash """
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except:
        print("Current codebase does not take part of a Git project...")
        sha = None
    return sha

#####===== Logging Functions =====#####

@log_function
def log_architecture(model: nn.Module, save_path: str):
    """
    Printing architecture modules into a txt file
    """
    assert save_path[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"

    # getting all_params
    with open(save_path, "w") as f:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total Params: {num_params}")

    for i, layer in enumerate(model.children()):
        if(isinstance(layer, torch.nn.Module)):
            log_module(module=layer, save_path=save_path)
    return


def log_module(module, save_path, append=True):
    """
    Printing architecture modules into a txt file
    """
    assert save_path[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"

    # writing from scratch or appending to existing file
    if (append is False):
        with open(save_path, "w") as f:
            f.write("")
    else:
        with open(save_path, "a") as f:
            f.write("\n\n")

    # writing info
    with open(save_path, "a") as f:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        f.write(f"Params: {num_params}")
        f.write("\n")
        f.write(str(module))
    return

@log_function
def save_checkpoint(model, optimizer, scheduler, epoch, save_path, finished=False, savename=None):
    """
    Saving a checkpoint in the models directory of the experiment. This checkpoint
    contains state_dicts for the mode, optimizer and lr_scheduler
    Args:
    -----
    model: torch Module
        model to be saved to a .pth file
    optimizer, scheduler: torch Optim
        modules corresponding to the parameter optimizer and lr-scheduler
    epoch: integer
        current epoch number
    exp_path: string
        path to the root directory of the experiment
    finished: boolean
        if True, current checkpoint corresponds to the finally trained model
    """

    if(savename is not None):
        checkpoint_name = savename
    elif(savename is None and finished is True):
        checkpoint_name = "checkpoint_epoch_final.pth"
    else:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"

    savepath = os.path.join(save_path, checkpoint_name)

    scheduler_data = "" if scheduler is None else scheduler.state_dict()
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scheduler_state_dict": scheduler_data
            }, savepath)
    print_(f'Checkpoint was saved to: {savepath}')

    return


#####===== Logger Modules =====#####

class Logger(object):
    def __init__(self, exp_name: Optional[str] = None, run_name: Optional[str] = None, exp_path: Optional[str] = None):
        assert (exp_name is not None and run_name is not None) or exp_path is not None, "ERROR: Please provide either an experiment and run name or an experiment path"
        self.exp_name = exp_name
        self.run_name = run_name
        self.exp_path = exp_path
        self.base_path = P('experiments') if P(exp_path) is None else exp_path

        ##-- Logging Paths --##
        if self.exp_name is not None and self.run_name is not None:
            self.run_path = self.base_path / self.exp_name / self.run_name
        else:
            self.run_path = self.base_path
        self.plot_path = self.run_path / "plots" 
        if not os.path.exists(str(self.plot_path)):
            os.makedirs(str(self.plot_path))
        self.log_path = self.run_path / "logs"
        if not os.path.exists(str(self.log_path)):
            os.makedirs(str(self.log_path))
        self.vis_path = self.run_path / "visualizations" 
        if not os.path.exists(str(self.vis_path)):
            os.makedirs(str(self.vis_path))
        self.checkpoint_path = self.run_path / "checkpoints"
        if not os.path.exists(str(self.checkpoint_path)):
            os.makedirs(str(self.checkpoint_path))
        self.log_file_path = self.log_path / 'log.txt'
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)

        ##-- Writer Modules --##
        self.csv_writer = None
        self.wandb_writer = None
        self.tb_writer = None
        self.internal_writer = None

        global LOGGER
        LOGGER = self
    
    def log(self, data: Dict[str, Any], step: Optional[int]=None) -> bool:
        if self.csv_writer is not None:
            self.csv_writer.log(data, step)
        if self.wandb_writer is not None:
            self.wandb_writer.log(data, step)
        if self.tb_writer is not None:
            self.tb_writer.log(data, step)
        if self.internal_writer is not None:
            self.internal_writer.log(data, step)

    def log_info(self, message: str, message_type: str='info') -> None:
        if not self.run_initialized:
            return
        cur_time = self._get_datetime()
        msg_str = f'{cur_time}   [{message_type}]: {message}\n'
        with open(self.log_file_path, 'a') as f:
            f.write(msg_str)  

    def log_config(self, config: Dict[str, Any]) -> None:
        """
            Log configuration to the log file.
        """

        cur_time = self._get_datetime()
        msg_str = f'{cur_time}   [config]:\n'
        msg_str += '\n'.join([f'  {k}: {v}' for k,v in config.items()])

        with open(self.log_file_path, 'a') as f:
            f.write(msg_str)

        if self.wandb_writer is not None:
            self.wandb_writer.log_config(config)

    def initialize_csv(self, file_name: str = None):
        file_name = P(self.log_path) / file_name if file_name is not None else P(self.log_path) / "metrics.csv"
        self.csv_writer = CSVWriter(file_name)
    
    def initialize_wandb(self, project_name: str, **kwargs):
        if self.exp_name is not None and self.run_name is not None:
            name = f"{self.exp_name}/{self.run_name}"
        else:
            name = self.run_path.stem
        self.wandb_writer = WandBWriter(name, project_name, **kwargs)

    def initialize_tensorboard(self, **kwargs):
        raise NotImplementedError('Tensorboard is not implemented yet')
    
    def initialize_internal(self, **kwargs):
        self.internal_writer = MetricTracker(**kwargs)
    
    def get_path(self, name: Optional[Literal['log', 'plot', 'checkpoint', 'visualization']] = None) -> str:
        """
            Get the path to the specified directory.

            Arguments:
                name Optional[Literal['log', 'plot', 'checkpoint', 'visualization']]: Type of the directory to retrieve.
                    If None is provided, the run path is returned.


        """
        if not self.run_initialized:
            return
        assert name in ['log', 'plot', 'checkpoint', 'visualization'], "Please provide a valid directory type"
        if name is None:
            return self.run_path
        if name == 'log':
            return self.log_path
        elif name == 'plot':
            return self.plot_path
        elif name == 'checkpoint':
            return self.checkpoint_path
        elif name == 'visualization':
            return self.vis_path
        
    def _get_datetime(self) -> str:
        return datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

class CSVWriter(object):
    """
        A small module to dynamically log metrics to a csv file.
    """
    def __init__(self, file_name: str, overwrite: Optional[bool] = True):
        if os.path.exists(file_name) and overwrite is False:
            i = 1
            file_name = P(file_name)
            file_dir = file_name.parent
            raw_name = file_name.stem
            while os.path.exists(str(file_dir / (raw_name+f"-{i}.csv"))):
                if i == 100:
                    print_(f"ERROR: CSV Writer, too many log files exist, please override...")
                    return
                i += 1
            file_name = str(file_dir / (raw_name+f"-{i}.csv"))
        self.file_name = file_name
        self.tracked_metrics = {"step": 0}

        with open(self.file_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.tracked_metrics.keys())

    def log(self, data: Dict[str, Any], step: Optional[int]=None) -> None:

        data_to_write = [step]+[None]*(len(self.tracked_metrics)-1)
        col_to_update = []

        for key, value in data.items():
            if key not in self.tracked_metrics:
                col_to_update.append(key)
            else:
                data_to_write[self.tracked_metrics[key]] = value
        
        if len(col_to_update) > 0:
            try:
                self.update_file(col_to_update)
            except Exception as e:
                print_(f"ERROR: CSV Writer, unable to update file: \n{e}")
                return False
            for name in col_to_update:
                data_to_write[self.tracked_metrics[name]] = data[name]
        try:
            with open(self.file_name, 'w') as csvfile:
                writer = csv.Writer(csvfile)
                writer.writerow(data_to_write)
        except Exception as e:
            print_(f"ERROR: CSV Writer, unable to write to file: {e}")
            return False
        return True

    def update_file(self, new_columns: List[str]):
        with open(self.file_name, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            # Read the existing content
            data = list(csv_reader)
            header = data[0]

        for i, name in enumerate(new_columns):
            self.tracked_metrics[name] = len(header)+i+1
        header += new_columns
        data[0] = header

        with open(self.file_name, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(data)

class MetricTracker:
    """
        A module to log metrics in RAM to easily access them and compute statistics.
    """
    def __init__(self):
        self.metrics = {}

    def log(self, data: Dict[str, Any]):
        """ Log a metric value"""
        for metric_name, metric_value in data.items():
            if metric_name not in self.metrics.keys():
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(metric_value)

    def get_metric(self, metric_name: Optional[str] = None):
        if metric_name is None:
            return self.metrics
        elif metric_name in self.metrics.keys():
            return self.metrics[metric_name]
        else:
            print_('Tried to fetch non-existing from MetricTracker', 'warn')
            return None

    def get_mean(self, metric_name: str = None) -> float:
        """ Get the mean value of a metric """
        if metric_name is None:
            return {key: np.mean(values) for key, values in self.metrics.items()}
        if metric_name not in self.metrics.keys():
            print_(f'MetricTracker received an invalid metric name for retrieval {metric_name}')
            return 0
        return np.mean(self.metrics[metric_name])

    def get_variance(self, metric_name: str = None) -> float:
        """ Get the variance value of a metric """
        if metric_name is None:
            return {key: np.var(values) for key, values in self.metrics.items()}
        if metric_name not in self.metrics.keys():
            print_(f'MetricTracker received an invalid metric name for retrieval {metric_name}')
            return 0
        return np.var(self.metrics[metric_name])
    
    def get_median(self, metric_name: str = None) -> float:
        """ Get the median value of a metric """
        if metric_name is None:
            return {key: np.median(values) for key, values in self.metrics.items()}
        if metric_name not in self.metrics.keys():
            print_(f'MetricTracker received an invalid metric name for retrieval {metric_name}')
            return 0
        return np.median(self.metrics[metric_name])

    def reset(self, metric_name: str = None):
        """
            Reset a metric. If no metric name is provided, all metrics are resetted.
        """
        if metric_name is not None:
            self.metrics[metric_name] = []
        else:
            self.metrics = {}

class TensorBoardWriter(object):
    """
        TODO: Finish this writer
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        return

class WandBWriter(object):

    def __init__(self, run_name: str, project_name: str, **kwargs):
        import wandb
        wandb.login()
        self.run = wandb.init(project=project_name, name=run_name, **kwargs)

    def log(self, data: Dict[str, Any], step: Optional[int]=None) -> bool:
        """
            Log data to WandB.

            Arguments:
                data [Dict[str, Any]]: Data to log of form: {metric_name: value, metric_name2: value2,...}

        """
        
        try:
            wandb.log(data, step)
        except Exception as e:
            print_('Logging failed: ', e)
            return False
        return True
    
    def log_config(self, config: Dict[str, Any]) -> None:
        self.run.config.update(config)
    
    def log_image(self, name: str, image: Union[torch.Tensor, np.array], step: Optional[int]=None) -> None:
        """
            Log images to WandB.
        """
        # import ipdb; ipdb.set_trace()
        assert len(image.shape) in [3, 4], "Please provide images of shape [H, W, C], [B, H, W, C], [C, H, W] or [B, C, H, W]"
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        if image.shape[-1] not in [1, 3]:
            if len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            elif len(image.shape) == 4:
                image = np.transpose(image, (0, 2, 3, 1))
        try:
            torchvision.utils.save_image(image, self.vis_path / f"{name}.png")
        except: 
            print_(f"Failed to save image {name} to {self.vis_path}.")
        wandbImage = wandb.Image(image)
        wandb.log({name: wandbImage}, step=step)
    
    def log_segmentation_image(self, name: str,
                  image: Union[torch.Tensor, np.array],
                   segmentation: Optional[Union[torch.Tensor, np.array]],
                    ground_truth_segmentation: Optional[Union[torch.Tensor, np.array]]=None,
                     class_labels: Optional[list] = None,
                      step: Optional[int]=None) -> None:
        """
            Log a segmentation image to WandB.

            Arguments:
                image [Union[torch.Tensor, np.array]]: Image to log.

        """
        assert len(image.shape) in [3, 4], "Please provide images of shape [H, W, C], [B, H, W, C], [C, H, W] or [B, C, H, W]"
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        if image.shape[-1] not in [1, 3]:
            if len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            elif len(image.shape) == 4:
                image = np.transpose(image, (0, 2, 3, 1))
        if torch.is_tensor(segmentation):
            segmentation = segmentation.detach().cpu().numpy()
        if ground_truth_segmentation is not None:
            if class_labels is not None:
                wandbImage = wandb.Image(image, masks={
                    "predictions": {
                        "mask_data": segmentation,
                        "class_labels": class_labels
                    },
                    "ground_truth": {
                        "mask_data": ground_truth_segmentation,
                        "class_labels": class_labels
                    }
                    })
            else:
                wandbImage = wandb.Image(image, masks={
                    "predictions": {
                        "mask_data": segmentation,
                    },
                    "ground_truth": {
                        "mask_data": ground_truth_segmentation,
                    }
                    })
        else:
            if class_labels is not None:
                wandbImage = wandb.Image(image, masks={
                        "predictions": {
                            "mask_data": segmentation,
                            "class_labels": class_labels
                        }})
            else:
                wandbImage = wandb.Image(image, masks={
                        "predictions": {
                            "mask_data": segmentation,
                        }})              
        wandb.log({name: wandbImage}, step=step)

    
LOGGER = None