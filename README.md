# Experiment Template
This is a template directory structure for experiments with deep learning models using PyTorch.
It is intended to serve as the base for a new project. It contains basic functions to handle the experiment directory structure, load/save models, load/save configs and perform some high-level debugging.
For me personally, the most important functionality is the logging logic that I use in all my projects.

A simple example project to see how this project template can be employed can be seen in the [example branch](../../tree/example)

## Structure
    .
    ├── config                          # Directory holding config files used for the experiments
    ├── data                            # Directory holding all data 
    ├── experiments                     # Directory holding all data related to the experiments
    │   └── example_exp                 # An example experiments directory
    │       └── example_run             # An example run directory
    │           ├── checkpoints         # Directory holding the checkpoints
    │           ├── logs                # Directory holding the logs (e.g. TensorBoard)
    │           ├── visualizations      # Directory holding visualizations
    │           ├── plots               # Directory holding plots
    │           └── config.json         # Config file for the run
    ├── src                             # Directory holding the complete source code
    │    ├── data                       # Directory to hold data loaders and utils functions for your data
    │    ├── models                     # Directory holding all source files for the models
    │    │   ├── util_modules.py        # Some useful torch modules
    │    ├── training                   # Directory to hold training functions 
    │    ├── utils                      # Directory holding utility functions for the project
    │    │   ├── helper_functions.py    # Some basic helper functions
    │    │   ├── logging.py             # Directory holding the complete logging logic
    │    │   ├── management.py          # Management functions for the experiment environment
    │    │   ├── losses.py              # Additional custom loss functions
    │    │   ├── model_debugger.py      # A debugger for PyTorch modules that hooks into the forward or backward pass
    │    │   └── profiler.py            # A profiler timing computation steps (currently only synchronous CPU operations)
    │    ├── visualization              # Directory holding the visualization functions
    │    │   ├── helper_functions.py    # General helper functions for visualization
    │    │   ├── visualizations.py      # Visualization functions
    │    │   └── plotting.py            # Functions used for plotting
    │    └── 01_initialize.py           # Initialize a new experiment and/or run
    ├── env.sh                          # Source file for the environment setup
    └── keep_exp.sh                     # File to create .gitkeep files in all directories
    
## Logging
The logging is completely encapsulated in the file `src/utils/logging.py`. It provides different modules and functions that can be individually used.
The most important part is the `Logger` class which creates a global logger module `LOGGER` that can be used throughout the project. 
This module provides different functionalities to log your training progress that can be activated individually through their respective intialization functions:
 * **W&B**: The W&B writer allows to log your data to the W&B servers.
 * **TensorBoard**: The TensorBoard writer writes your logs to local files which can be viewed through a local TensorBoard server.
 * **CSV**: The CSV writer writes all logs to a CSV file to be easily imported or viewed through programs such as Microsoft Excel or LibreOffice Calc.

### Workflow
Here, I describe how I typically use the logging module. An in-practice example can be seen in the [example branch](../../tree/example).

1. In your training function first initialize the logger: `logger = Logger(exp_name, run_name)`
2. Activate the writers you want to use: `logger.initialize_csv() \\ logger.initialize_wandb('example')`
3. Whenever you want to log something simply call: `logger.log({metric_name: metric_value}, step)`
4. Log some evaluation image: `logger.log_image(image_name, image)`
