"""
    This module capsules the logging to WandB servers, Tensorboard, or simply the local experiment directory.
    Ideally this should capsule the complete communication with the experiment directory etc.

    Some parts of the logging module were adapted from: https://github.com/angelvillar96/TemplaTorch

    Author: Luis Denninger <l_denninger@uni-bonn.de>

"""

import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union, Literal
import traceback
from datetime import datetime
import os
import git
from pathlib import Path as P
import csv
import wandb
import matplotlib.pyplot as plt

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
    """ Log a message to the log files of the logger. """
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
def save_checkpoint(epoch, model=None, optimizer=None, scheduler=None, save_path=None, finished=False, save_name=None):
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

    if(save_name is not None):
        checkpoint_name = save_name+f'epoch_{epoch}.pth'
    elif(save_name is None and finished is True):
        checkpoint_name = "checkpoint_epoch_final.pth"
    else:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
    if save_path is None:
        if LOGGER is not None:
            save_path = LOGGER.get_path('checkpoint')
        else:
            print_("Please provide a save path to save checkpoints", 'error')
            return False

    savepath = os.path.join(save_path, checkpoint_name)

    data = {'epoch': epoch}
    if model is not None:
        data['model_state_dict'] = model.state_dict()
    if optimizer is not None:
        data['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        data['scheduler_state_dict'] = scheduler.state_dict()

    try:
        torch.save(data, savepath)
    except Exception as e:
        print_(f"Could not save checkpoint to {savepath}. \n{e}", 'error')
        return False
    print_(f'Checkpoint was saved to: {savepath}')

    return True

#####===== Logger Modules =====#####

class Logger(object):
    """
        Main logging module. This should be always instantiated when training or evaluating a model.
        Through this module, you can then activate different writers to log the training or evaluation process.
        Available writers are: WandB writer, Tensorboard writer or CSV writer.
    
    """
    def __init__(self, exp_name: Optional[str] = None, run_name: Optional[str] = None, exp_path: Optional[str] = None):
        """
            Initialize the logger. This also initializes a global logger that can be accessed throughou the project.

            Arguments:
            -----------
            @param exp_name: Name of the experiment
            @param run_name: Name of the run
            @param exp_path: Path to the root directory of the experiments.
        """
        assert (exp_name is not None and run_name is not None) or exp_path is not None, "ERROR: Please provide either an experiment and run name or an experiment path"
        self.exp_name = exp_name
        self.run_name = run_name
        self.exp_path = exp_path
        self.base_path = P('experiments') if exp_path is None else P(exp_path)

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
    
    ##-- Logging Functions --##
    def log(self, data: Dict[str, Any], step: Optional[int]=None) -> None:
        """
            Log scalar metrics.

            Arguments:
            -----------
            @param data: Dictionary mapping metric_name->metric_value to be logged.
            @param step: Current training step.
        """
        if self.csv_writer is not None:
            self.csv_writer.log(data, step)
        if self.wandb_writer is not None:
            self.wandb_writer.log(data, step)
        if self.tb_writer is not None:
            for k, v in data.items():
                self.tb_writer.add_scalar(k, v, step)
        if self.internal_writer is not None:
            self.internal_writer.log(data, step)

    def log_info(self, message: str, message_type: str='info') -> None:
        """
            Log information to the log file.

            Arguments:
            -----------
            @param message: Message to be logged.
            @param message_type: Type of the message that is displayed in the log file.
        """
        cur_time = self._get_datetime()
        msg_str = f'{cur_time}   [{message_type}]: {message}\n'
        with open(self.log_file_path, 'a') as f:
            f.write(msg_str)  

    def log_config(self, config: Dict[str, Any]) -> None:
        """
            Log configuration to the log file.

            Arguments:
            -----------
            @param config: Config dictionary mapping config_name->config_value to be logged.
        """

        cur_time = self._get_datetime()
        msg_str = f'{cur_time}   [config]:\n'
        msg_str += '\n'.join([f'  {k}: {v}' for k,v in config.items()])

        with open(self.log_file_path, 'a') as f:
            f.write(msg_str)

        if self.wandb_writer is not None:
            self.wandb_writer.log_config(config)

    def log_architecture(self, model: torch.nn.Module) -> None:
        """ 
            Log the model architecture. 

            Arguments:
            -----------
            @param model:  PyTorch model to be logged.
        """
        savePath = str(P(self.log_path) / "architecture.txt")
        log_architecture(model, savePath)
    
    def log_git_hash(self) -> None:
        """ Log the git hash from the current commit. """
        gitHash = get_current_git_hash()
        with open(self.log_path / 'git_hash.txt', 'w') as f:
            f.write(gitHash)

    def log_image(self, name: str, image: Union[torch.tensor, np.array], step: Optional[int] = None) -> None:
        """ 
            Log an image to the experiment directory.

            Arguments:
            -----------
            @param name: Name of the image that is used to save the file.
            @param image: Image to be logged. The image should be of format [height,width,channel]
            @param step: Current training step.
        """
        savePath = str(P(self.vis_path) / name)
        plt.imsave(savePath, image)
        if self.wandb_writer is not None:
            self.wandb_writer.log_image(name, image, step)
        if self.tb_writer is not None and False:
            self.tb_writer.log_image(name, image, step)

    def log_histograms(self, name: str, values: Union[np.array, torch.tensor]):
        """
            Log a histogram to the experiment directory.

            Arguments:
            -----------
            @param name: Name of the histogram that is used to save the file.
            @param values: Values to be logged.

            TODO: This function was not properly tested yet and should be used with caution
        """
        if torch.is_tensor(values):
            values = values.cpu().numpy()
        if self.wandb_writer is not None:
            self.wandb_writer.log_histogram(name, values)

    def save_checkpoint(self,  epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer=None, scheduler=None):
        """
            Save a model checkpoint during training. 
            Optionally, one can also provide the optimization modules to resume training
            from the given checkpoint.

            Arguments:
            -----------
            @param epoch: Current training epoch.
            @param model: PyTorch model to be saved.
            @param optimizer: PyTorch optimizer to be saved.
            @param scheduler: PyTorch scheduler to be saved.
        """

        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            save_path=self.checkpoint_path
        )


    ##-- Initialization Functions --##
    # This functions have to be called to activate the different writers.

    def initialize_csv(self, file_name: str = None):
        """
            Initialize the CSV writer. The file is saved to the log path of the run.

            Arguments:
            -----------
            @param file_name: Name of the CSV file to be saved. 
        """
        file_name = P(self.log_path) / file_name if file_name is not None else P(self.log_path) / "metrics.csv"
        self.csv_writer = CSVWriter(file_name)
    
    def initialize_wandb(self, project_name: str, **kwargs):
        """
            Initialize the WandB writer. This requires prior logging to your account through your API key.

            Arguments:
            -----------
            @param project_name: Name of the project within the WandB environment.
            @param kwargs: Additional arguments according to the WandB module: https://docs.wandb.ai/ref/python/init
        """
        if self.exp_name is not None and self.run_name is not None:
            name = f"{self.exp_name}/{self.run_name}"
        else:
            name = self.run_path.stem
        self.wandb_writer = WandBWriter(name, project_name, **kwargs)

    def initialize_tensorboard(self, **kwargs):
        """
            Initialize the TensorBoard writer.

            Arguments:
            -----------
            @param kwargs: Additional arguments according to the TensorBoard module.
        """
        self.tb_writer = TensorboardWriter(self.log_path, **kwargs)
    
    
    def get_path(self, name: Optional[Literal['log', 'plot', 'checkpoint', 'visualization']] = None) -> str:
        """
            Get the path to the specified directory.

            Arguments:
            -----------
            @param name: Type of the directory to retrieve the path from.
                The run directory consists of the following specific directories: 
                - log -> To save any logs
                - plot -> To save any plots
                - checkpoint -> To save model checkpoints
                - visualization -> To save any visualizations

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
        """ Internal function to get the current formatted time. """
        return datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

class CSVWriter(object):
    """
        A small module to dynamically log metrics to a csv file.
    """
    def __init__(self, file_name: str, overwrite: Optional[bool] = True):
        """
            Initialize the CSW writer.

            Arguments:
            -----------
            @param file_name: Name of the CSV file to be saved.
            @param overwrite: Whether to overwrite the file if it already exists.
        """
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
        """
            Log the given metrics.
            The function automatically adds new columns for each metric.

            Arguments:
            -----------
            @param data: Metrics to be logged. The dictionary is of format {metric_name: metric_value}.
            @param step: Current training step.
        """

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
            data_to_write += [None] * (len(self.tracked_metrics) - len(data_to_write)) 
            
            for name in col_to_update:
                data_to_write[self.tracked_metrics[name]] = data[name]
        try:
            with open(self.file_name, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_to_write)
        except Exception as e:
            print_(f"ERROR: CSV Writer, unable to write to file: {e}")
            return False
        return True

    def update_file(self, new_columns: List[str]) -> None:
        """
            Internal function to update the columns with newly tracked metrics.

            Arguments:
            -----------
            @param new_columns: List of new columns (names of the metrics) to be added.
        """
        with open(self.file_name, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            # Read the existing content
            data = list(csv_reader)
            header = data[0]

        for i, name in enumerate(new_columns):
            self.tracked_metrics[name] = len(header)+i
        header += new_columns
        data[0] = header

        with open(self.file_name, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(data)

class TensorboardWriter:
    """
    Class for handling the tensorboard logger

    """

    def __init__(self, logdir: str) -> None:
        """ 
            Initializing tensorboard writer.

            Arguments:
            -----------
            @param logdir: Path to the log directory.
        """
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        return

    def add_scalar(self, name:str, val:Union[float,int], step:int):
        """ 
            Adding a scalar for plot.

            Arguments:
            -----------
            @param name: Name of the scalar.
            @param val: Value of the scalar.
            @param step: Current training step.
        """
        self.writer.add_scalar(name, val, step)
        return

    def add_scalars(self, plot_name, val_names, vals, step):
        """ Adding several values in one plot """
        val_dict = {val_name: val for (val_name, val) in zip(val_names, vals)}
        self.writer.add_scalars(plot_name, val_dict, step)
        return

    def add_image(self, fig_name, img_grid, step):
        """ Adding a new step image to a figure """
        self.writer.add_image(fig_name, img_grid, global_step=step)
        return

    def add_figure(self, tag, figure, step):
        """ Adding a whole new figure to the tensorboard """
        self.writer.add_figure(tag=tag, figure=figure, global_step=step)
        return

    def add_graph(self, model, input):
        """ Logging model graph to tensorboard """
        self.writer.add_graph(model, input_to_model=input)
        return

    def log_full_dictionary(self, dict, step, plot_name="Losses", dir=None):
        """
        Logging a bunch of losses into the Tensorboard. Logging each of them into
        its independent plot and into a joined plot
        """
        if dir is not None:
            dict = {f"{dir}/{key}": val for key, val in dict.items()}
        else:
            dict = {key: val for key, val in dict.items()}

        for key, val in dict.items():
            self.add_scalar(name=key, val=val, step=step)

        plot_name = f"{dir}/{plot_name}" if dir is not None else key
        self.add_scalars(plot_name=plot_name, val_names=dict.keys(), vals=dict.values(), step=step)
        return

class WandBWriter(object):
    """
        The WandB writer encapsulating the communication with the WandB servers.
    """

    def __init__(self, run_name: str, project_name: str, **kwargs):
        """
            Initialize the writer.

            Arguments:
            -----------
            @param run_name: Name of the run.
            @param project_name: Name of the project.
            @param kwargs: Additional arguments to be passed to wandb.init: https://docs.wandb.ai/ref/python/init
        """
        wandb.login()
        self.run = wandb.init(project=project_name, name=run_name, **kwargs)

    def log(self, data: Dict[str, Any], step: Optional[int]=None) -> bool:
        """
            Log data to WandB.

            Arguments:
            -----------
            @param data: Metrics to be logged. The dictionary is of format {metric_name: metric_value}.
            @param step: Current training step.

        """
        
        try:
            wandb.log(data, step)
        except Exception as e:
            print_('Logging failed: ', e)
            return False
        return True
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
            Log the configuration parameters.

            Arguments:
            -----------
            @param config: Configuration parameters to be logged. The dictionary is of format {config_name: config_name}.
        """
        self.run.config.update(config)

    def log_histogram(self, name: str, values: Union[torch.Tensor, np.array], step: Optional[int]=None) -> None:
        """
            Log a histogram to WandB.

            Arguments:
            -----------
            @param name: Name of the histogram.
            @param values: Values of the histogram.
            @param step: Current training step.

            TODO: This function was not properly tested yet and should be used with caution
        """
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()

        hist = wandb.Histogram(values)
        wandb.log({name: hist}, step=step)
    
    def log_image(self, name: str, image: Union[torch.Tensor, np.array], step: Optional[int]=None) -> None:
        """
            Log images to WandB. Images should be given in RGB format.

            Arguments:
            -----------
            @param name: Name of the image.
            @param image: Image to be logged. Image can be given in formats: [H, W, C] or [C, H, W] 
            @param step: Current training step.

            TODO: Add batched image support
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

            TODO: !! This function does not work right now !!

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

class MetricTracker(object):
    """ 
        Manages and stores different metrics.
        This can be used to efficiently track the sum and average of a metric over the course of an epoch.
    """

    def __init__(self) -> None:
        """ Initialize the metric tracker """
        self.metrics = None
        self.reset()

    def reset(self):
        """ Deleted and reset all metrics. """
        self.metrics = {}

    def update(self, name:str, val:Union[float,int, List[Union[float,int]]])->None:
        """
            Update the tracked metrics.

            Arguments:
            -----------
            @param name: Name of the metric.
            @param val: Value of the metric. Metrics can be given as scalars or a list of scalars
        """
        if name not in self.metrics:
            i = 1 if type(val)!=list else len(val)
            self.metrics[name] = AverageMeter(i)
        self.metrics[name].update(val)

    def get_average(self, name:str=None) -> Union[Dict[str, List[float]], List[float]]:
        """
            Retrieve the average of the tracked metrics.
            If a name is provided it only returns the given metric, else all metrics are returned.

            Arguments:
            -----------
            @param name: Name of the metric. If None all metrics are returned.

            @return Dict[str, List[float]] or List[float]: Average of the given metrics
        """
        if name is None:
            ret_dict = {}
            for name, meter in self.metrics.items():
                ret_dict[name] = meter.avg
            return ret_dict
        return self.metrics[name].avg
    
    def get_sum(self, name:str=None) -> Union[Dict[str,List[float]],List[float]]:
        """
            Retrieve the sum of the tracked metrics.
            If a name is provided it only returns the given metric, else all metrics are returned.

            Arguments:
            -----------
            @param name: Name of the metric. If None all metrics are returned.

            @return Dict[str, List[float]] or List[float]: Sum of the given metrics
        """
        if name is None:
            ret_dict = {}
            for name, meter in self.metrics.items():
                ret_dict[name] = meter.sum
            return ret_dict
        return self.metrics[name].sum



class AverageMeter(object):
    """Computes and stores the average, sum and current value"""

    def __init__(self, i:int=1, precision:int=3):
        """
            Initialize the AverageMeters.

            Arguments:
            -----------
            @param i: Number of different metrics saved.
            @param precision: Precision of the average at print out.
        """
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i:int=None):
        """
            Reset the average meters.

            Arguments:
            -----------
            @param i: Number of different metrics saved. If not provided the number given at initialization is used
        """
        if i is None:
            i = self.meters
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.count = 0

    def update(self, val:Union[float,int], n:int=1):
        """
            Update the average meters.

            Arguments:
            -----------
            @param val: Value of the metric. Metrics can be given as scalars or a list of scalars.
            @param n: Number of times the metric is updated.
        """
        if not isinstance(val, list):
            val = [val]
        assert(len(val) == self.meters)
        self.count += n
        for i,v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        """ Representation overload to show average and value."""
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)
# Global logger
LOGGER = None