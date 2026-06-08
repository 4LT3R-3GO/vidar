# Readme-file


## Overview
This project implements a reinforcement learning agent to generate SQL queries for data extraction.
This repository contains the training code, evaluation tools, pretrained models, and evaluation logs used in the project report.

The Python environment used for this project is `3.13.2`.

## Project Structure

## Installation Instructions

### 1. Clone the repository
If the artifact is delivered as a zip file, unzip it and continue to the next section.

Ensure to change the directory to the desired work folder before cloning the repository.
```bash
git clone https://github.com/4LT3R-3GO/BPR
cd BPR
```

### 2. Installation of required Python libraries

#### <i>Option 1: Using PIP</i>

The required Python libraries are listed in the requirements.txt file, as standardized. 

```bash
python -m venv env                 #Create a virtual environment
env\Scripts\activate               #For Windows. Linux and other OS use other filenames and paths
pip install -r requirements.txt    #Installing required libraries
```
#### <i>Option 2: Using uv (Recommended)</i>
uv can be used in two ways; either installing libraries using a faster version of PIP, or by upgrading to a uv-managed project. However, this requires uv to be installed, as it does not come as a standardized tool. 

<i>PIP</i>

```bash
uv pip install -r requirements.txt  
```

<i>uv-managed</i>

```bash
uv init                        #Initialize the environment, and create the pyproject.toml
uv add -r requirements.txt     #Add and install libraries listed in requirements.txt file.
```

<i>Installing uv</i>
If uv is not installed, it can be installed by either downloading and following the instructions at <a href>https://github.com/astral-sh/uv, or by installing with `pip install uv`. For convenience, the author published uv to PyP.


## Usage
This project is twofold, and therefore contains required tools and environments for both the creation of the database, the training of the model, and the testing and evaluation of the model.

The repository structure is displayed in [File inclusion tree](#file-inclusion-tree). This section of the readme will provide context and usage information for specific files. Please refer to the File tree for guidance on repository locations.

Training expects the SQLite database file to be available at `database/backend.db`
If this file is missing, the custom environment cannot be initialized. The database can be initialized by running `database/init.sql` with SQLite.

This project is not a standalone application. Configuration is performed by modifying variables directly in the Python scripts rather than using command-line arguments.

> **Disclaimer**  
> This repository does not include the required files or modification details for the online testing environment.  
This is due to bWAPP being a legacy application that requires PHP-specific modifications.  
> Additional setup details are described in the project paper.

### Python Scripts 
The repository's root folder contains two Python scripts: one for model training and one for model testing. Both of these scripts use utilities that are imported at runtime. It is important to run the scripts from the repository root, as utilities are imported using relative paths.  
These utilities reside in the `utils` folder at the root. 

#### Model Training
Training is started with:

```bash
python train.py
```

The training script uses the custom environment defined in `utils/CustomGym.py`.

The most important configuration values in `train.py` are:

- `STAGE_CONFIG`: hyperparameter configuration for each curriculum stage
- `UPDATES_TARGET`: number of policy updates to perform
- `EP_LENGTH`: episode length for each stage
- `LR_FLOOR_RATE_EP`: per-stage scaling factors used to define the minimum learning rate in the linear learning-rate schedule.
- `NUM_ENVS`: number of parallel environments used during training
- `parallellism`: selects whether to use `DummyVecEnv` (`"single"`) or `SubprocVecEnv` (`"multiple"`).
- `version_number`: version tag used in saved model filenames and TensorBoard run names.

The script is currently configured through direct variable editing in `train.py`. To run a different stage, change the stage loop in the `__main__` block and adjust the relevant stage parameters if needed. This is defined using a for-loop (i.e., `for s in range(1,6):  # selects which training stages to run`), where s is used to define the current stage, passed to the environment to adjust both the initial setup and reward system. 

During training, an evaluation callback periodically saves the best-performing model.  
By default, the evaluation frequency is set to `n_steps // 2`.


#### Model Evaluation

Evaluation is started with:

```bash
python eval.py
```

The evaluation script loads a trained `RecurrentPPO` model and runs it against a live target environment using `utils/EvalGym.py`.

Before running `eval.py`, the following configuration values must be reviewed and updated inside the script:

- `MODEL_PATH`: Path to the trained model that should be evaluated.
- `TARGET_URL`: URL of the target application endpoint used during evaluation.
- `TARGET_PARAM`: Name of the HTTP parameter that receives the generated payload.
- `COOKIES`: Session and security-related cookies required to access the target application.
- `EPISODE_LENGTH`: Maximum number of interaction steps allowed during the evaluation episode.

The evaluation environment is initialized through `SqlEvaluationGym`, after which the trained model is loaded and used to predict actions step by step until the episode ends or is truncated.


> **The script currently assumes:**
> - A trained model exists at `model/final_model.zip`
> - The target is reachable from the host machine
> - The provided cookies are valid (if required)



## File inclusion tree

 
```bash
|-- database            #Folder containing files related to the SQLite database. 
|   |-- backend.db      #Database file.
|   |-- init.sql        #Initiation file for creating and populating the SQLite database.
|   |-- nuke.sql        #File for removing the tables from the database.
|   \-- sqlite_tutorial.md  #SQLite Windows installation guide. 
|-- env [...]           #Python virtual environment folder, which is not mirrored to the repository.
|-- model               #Folder for the model
|   |-- backup [...]    #Empty folder, used as backup storage for models. 
|   |-- best_models [...]  #Empty folder, callback save folder.
|   |-- eval_logs [...] #Folder used for storing the stage evaluation logs.
|   |-- logs
|   |   \-- evaluations.npz
|   \-- final_model.zip #Pre-trained model
|-- Statistics_eval [...] #Folder used for extraction of TensorBoard data and plotted using matplotlib
|-- tb_logs [...]       #Folder for storing Tensorboard logs.
|-- tb_logs_archive [...]  #Folder for archiving old Tensorboard logs.
|-- tmp [...]           #Folder for temporary storage
|-- utils               #Folder for utilities 
|   |-- __pycache__ [...]
|   |-- __init__.py
|   |-- actions.py      #Helper file containing action space, HTTP response categories, and dialect translation
|   |-- callbacks.py    #Script containing customized callbacks from StableBaselines 3
|   |-- CustomGym.py    #Custom gym environment for the model training
|   |-- DatabaseAux.py  #Helper file for local SQLite database interaction
|   |-- error_msg.py    #File with list of expected error messages from SQLite
|   \-- EvalGym.py      #Custom gym environment for the online model evaluation  
|-- .gitignore
|-- eval.py             #Script for initiating online model evaluation
|-- README.md
|-- requirements.txt
|-- train.py            #Script for initiating model training
```

## Author
Michael Vaagland