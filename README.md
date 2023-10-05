# Experiment Template
This is a template directory structure for experiments with deep learning models using PyTorch.
It is intended to serve as the base for a new project. It contains basic functions to handle the experiment directory structure, load/save models, load/save configs and perform some high-level debugging.

## Structure
    .
    ├── config                      # Directory holding config files used for the experiments
    ├── data                        # Directory holding all data 
    ├── experiments                 # Directory holding all data related to the experiments
    │   └── exp_1                   # An example experiments directory
    │       ├── run_1               # An example run directory
    │           ├── checkpoints     # Directory holding the checkpoints
    │           ├── logs            # Directory holding the logs (e.g. TensorBoard)
    │           ├── visualizations  # Directory holding visualizations
    │           └── config.json     # Config file for the run
    ├── src                         # Directory holding the complete source code
    │    ├── models                 # Directory holding all source files for the models
    │    │   ├── util_modules.py    # Some useful torch modules
    │    ├── utils                  # Directory holding utility functions for the project
    │    │   ├── helper_functions.py# Some basic helper functions
    │    │   ├── management.py      # Management functions for the experiment environment
    │    │   ├── model_debugger.py  # A debugger for PyTorch modules that hooks into the forward or backward pass
    │    │   └── profiler.py        # A profiler timing computation steps (currently only synchronous CPU operations)
    │    ├── visualization          # Directory holding the visualization functions
    │    │   └── plotting_utils.py  # Some helper functions for visualization
    ├── env.sh                      # Source file for the environment setup
    ├── keep_exp.sh                 # File to create .gitkeep files in all directories
    └── run.py                      # The main file to run all functions
    
## Experiment Management
The project is designed such that the basic operations such as training or evaluation can be run through the file `run.py`.
The file `env.sh` gives basic aliases for quick configuration of the workspace and running of functions. <br/>

Initially please run: `source env.sh` <br/>
Current experiment and run name are stored as environment variables, such that we can easily manipulate and work with a run. 
If the environment variables are set, one can ommit specifying the experiment or run. <br/>
Show current experiment setup: `setup` <br/>
Set experiment name: `setexp [exp. name]` <br/>
Set run name: `setrun [run name]` <br/>
Initialize new experiment: `iexp -exp [exp. name]` <br/>
Initialize new run: `irun -exp [exp. name] -run [run name]` <br/>
Clear tensorboard logs from run: `cllog -exp [exp. name] -run [run name]` <br/>
Train a model: `train -exp [exp. name] -run [run name]` <br/>
