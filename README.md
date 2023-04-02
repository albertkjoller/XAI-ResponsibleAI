Project on XAI (Explainable AI)
==============================

Special course in Responsible AI @ DTU


## Setup

### Create environment
Clone the repository and create a virtual environment (with Python 3.10). A pre-defined environment running with CUDA 11.6 can be created like:

Run the following:

```
conda create -n xai_project python=3.10
conda activate xai_project
```

Install the dependencies:
```
pip install -r requirements.txt
```

### Data

Download the data with `dvc`:

``` 
dvc pull
```

#### PyTorch - CPU
If running on CPU install Pytorch with the following command:

```
pip3 install torch torchvision torchaudio
```

#### PyTorch - GPU (CUDA 11.6)
If running on GPU with CUDA 11.6 install Pytorch with the following command:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Using the dataloader to set up the data
To set up the required files for the training to run, run the ```bottleneck_code/data_processing.py``` file providing both, the data dir and the saving dir. An example can be seen bellow:
```
python ./src/data/bottleneck_code/data_processing.py -data_dir ./data/raw/CUB_200_2011 -save_dir ./data/processed/CUB_200_2011/bottleneck
```

## Using tensorboard in HPC
Run the following in your HPC terminal:
```
tensorboard --logdir logs --port 40000 --host $HOSTNAME
```
At the end of the response you get something like this: TensorBoard 2.10.1 at http://n-62-20-1:40000/ (Press CTRL+C to quit)

Afterwards, run in your local one:
```
ssh USER@l1.hpc.dtu.dk -g -L8080:n-62-20-1:40000 -N
```
Open in your browser: http://localhost:8080/
## Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── catalan_data_course            <- The original, immutable data dump.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   |   ├── __init__.py
    │   │   └── dataloader.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── __init__.py
    │       ├── model.py
    │       └── train_model.py
    │
    └── requirements.txt 
