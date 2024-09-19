# Network reconstruction may not mean dynamics prediction



## Overview

- This repository provides the code for the framework presented in [Network reconstruction may not mean dynamics prediction](https://arxiv.org/abs/2409.04240):
- Inferring latent dynamics equations from complex time series is a fundamental task of studying complex systems, including cellular systems, ecological systems, economic systems, climate pattern dynamics, disease spreading dynamics, and brain dynamics. We ask an interesting question of whether a good network reconstruction can help to predict future time series of complex systems, which has wide applicability in the studies of complex
system dynamics.
- In this manuscript, we provide a solid quantitative criteria to determine the quality of dynamics forecasting based on inferred network structures. This criteria is derived from a theoretical model of next-step prediction principle and associated dynamical mean-field theory. The criteria clarifies a transition point before which the dynamics and the network latent structure can be both well predicted, but after which the dynamics can only be forecasted up to some short time point even if the network structure can be inferred with a high accuracy (but not perfectly recovered). This theoretical finding is collaborated by our numerical simulation in concrete systems.
- Our study implies that it should be careful to draw a conclusion about future behavior of dynamically adaptive complex systems, especially high- dimensional systems such as brain circuits, even if a nearly-perfect construction of the circuit connectome is possible. This finding will thus yield broad impacts on the diverse fields of dynamics modeling and forecasting, either in a model-free or model-based context.


## Requirements

- Python 3.11
- PyTorch 2.3.0 (with CUDA 11.8)
- matplotlib-base 3.8.4
- numpy 1.24.3
- tqdm 4.66.2

## Installation
You can install these dependencies using conda:
- conda install pytorch==2.3.0 cudatoolkit=11.8 matplotlib-base==3.8.4 numpy==1.24.3 tqdm==4.66.2 -c pytorch


## Usage

### Experiments with RNN Learning Trajectories
- The code for different cases of `g` only differs in the hyperparameters.
- Run the code in numerical order as named.
- (`config.py`)` specifies parameter settings.
- (`1Training set and weight generator.py`)` generates weights and training datasets.
- (`2train.py`)` performs the neural network training process.
- (`3Weight map of maximum difference.py`)` compares the teacher's and student's weights.
- (`4Trajectory diagram.py`)` generates a diagram of the trajectories.
- (`5Trajectory Difference diagram.py`)` is used to show the quality of predictions.



#### The settings file (config.py) contains the following input arguments:
- `N`: The spatial dimension of the RNN network.
- `g`: The strength of the weights.
- `delta_t`: The discrete time step.
- `iterations`: The number of iteration time steps.
- `Truncate`: The number of time steps to truncate.
- `set_num`: The number of trajectories generated.
- `set_num_sect`: The number of trajectories selected for training.
- `chunk`: The number of small time steps into which truncated time steps are divided to create small samples.
- `batch_size`: The size of the batch.


### DMFT simulation and numerical correspondence
- The code (`total.py`) can be run directly.
- This section focuses on performing numerical simulations of DMFT and comparing direct dynamics. You can set the parameters $\eta$ and `g` for the simulations.

### The effect of DMFT in machine learning
- Except for the hyperparameters and `(4DMFT map in machine learning.py)`, the code structure in this section is the same as in Experiments with RNN Learning Trajectories.
- The code for different cases of `g` only differs in the hyperparameters.
- Run the code in numerical order as named.
- (`config.py`)` specifies parameter settings.
- (`1Training set and weight generator.py`)` generates weights and training datasets.
- (`2train.py`)` performs the neural network training process.
- (`3Weight map of maximum difference.py`)` compares the teacher's and student's weights.
- (`4DMFT map in machine learning.py`)` The $\eta$ calculated from the experimental results is used for DMFT simulation .
- (`5Trajectory diagram.py`)` generates a diagram of the trajectories.
#### The settings file (config.py) contains the following input arguments:
- `N`: The spatial dimension of the RNN network.
- `g`: The strength of the weights.
- `delta_t`: The discrete time step.
- `iterations`: The number of iteration time steps.
- `Truncate`: The number of time steps to truncate.
- `set_num`: The number of trajectories generated.
- `set_num_sect`: The number of trajectories selected for training.
- `chunk`: The number of small time steps into which truncated time steps are divided to create small samples.
- `batch_size`: The size of the batch.

### Classical Lyapunov exponent
- The code (`lledian.py`) can be run directly.
- Using the OS method to calculate the LLE of the classic RNN.

### dp figure
- The code (`dp.py`) can be run directly.
- Using the OS method to calculate the LLE of the our model.

## Citation
This code is the product of work carried out by the group of [PMI lab, Sun Yat-sen University](https://www.labxing.com/hphuang2018). If the code helps, consider giving us a shout-out in your publications.

## Contact
If you have any question, please contact me via yuzd610@163.com.












