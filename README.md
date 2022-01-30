#### Paper Title: A Sliding Window Based Approach With Majority Voting for Online Human Action Recognition using Spatial Temporal Graph Convolutional Neural Networks

# STGCN-SWMV

This is the code, data repository and pre-trained models for our work : **A Sliding Window Based Approach With Double Majority Voting for Online Human Action Recognition using Spatial Temporal Graph Convolutional Neural Networks**.

## Introduction

We propose to tackle the online action recognition problem with a sliding window and majority voting approach using Spatial Temporal Graph Convolutional Neural Networks.
Using only skeletal data, we first consider the 3D coordinates of each joint of the human skeleton as a chain of motion and represent them as a graph of nodes. The spatial and temporal evolution of this characteristic vector is then represented by a trajectory in the space of actions allowing us to simultaneously capture both geometrical appearances of the human body and its dynamics over time. The online action recognition problem is then formulated as a problem of finding similarities between the shapes of trajectories over time.

## Pre-requisites & Installation

Install all required libraries by running :

``` shell
pip install -r requirements.txt
cd torchlight & python setup.py install & cd ..
```

## Download data and pre-trained models
We provided the data and the pre-trained models of our **STGCN-SWMV** method for the OAD and UOW datasets. To download them, please run these scripts :
```
bash tools/rsc/get_data.sh
bash tools/rsc/get_weights.sh
```
**For Windows users :**

First, download WGet.exe from this link : [WGet](https://eternallybored.org/misc/wget/1.20.3/64/wget.exe) and copy it to the Windows/System32 directory.
Then open bash files with [GIT](https://git-scm.com/download/win).

## Results

Here are our results using the STGCN-SWMV method on the OAD and UOW online skeleton-based datasets.

**OAD:**
<p align="center">
	<img src="rsc/OAD Confusion Matrix.png" alt="OAD_Confusion_Matrix">
</p>

| Actions | Results | 
|:-------:|:-------:|
| Drinking | 0.979 |
| Eating | 1.000 |
| Writing | 0.993 |
| Opening cupboard | 1.000 |
| Washing hands | 1.000 |
| Opening microwave | 0.963 |
| Sweeping | 0.996 |
| Gargling | 0.984 |
| Trowing trash | 0.949 |
| Wiping | 0.973 |
| **Overall** | **0.954** |

**UOW:**
<p align="center">
	<img src="rsc/UOW Confusion Matrix.png" alt="UOW_Confusion_Matrix">
</p>

| Actions | Results | 
|:-------:|:-------:|
| High arm wave | 0.892 |
| Horizontal arm wave | 0.971 |
| Hammer | 1.000 |
| Hand catch | 0.857 |
| Forward punch | 0.929 |
| High Throw | 0.900 |
| Draw X | 0.969 |
| Draw Tick | 1.000 |
| Draw circle | 0.952 |
| Hand clap | 0.966 |
| Two Hand wave | 1.000 |
| Side boxing | 0.972 |
| Bend | 1.000 |
| Forward kick | 0.900 |
| Side kick | 1.000 |
| Jogging | 1.000 |
| Tennis swing | 0.974 |
| Tennis serve | 0.953 |
| Golf swing | 0.917 |
| Pick up and Throw | 0.917 |
| **Overall** | **0.934** |

## Test models

To test the **STGCN-SWMV** method and replicate our results, please run :

**For the OAD dataset :**

```python main.py stgcn_swmv --dataset=OAD --use_gpu=True -c config/stgcn_swmv/OAD/test.yaml```

**For the UOW dataset :**

```python main.py stgcn_swmv --dataset=UOW --use_gpu=True -c config/stgcn_swmv/UOW/test.yaml```

## Training from scratch

To train the **STGCN-SWMV** method from the scratch, please run :

**For the OAD dataset :**

```python main.py stgcn_swmv --dataset=OAD --use_gpu=True -c config/stgcn_swmv/OAD/train.yaml```

**For the UOW dataset :**

```python main.py stgcn_swmv --dataset=UOW --use_gpu=True -c config/stgcn_swmv/UOW/train.yaml```

**NOTE** : If --use_gpu is set to true make sure you have installed [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) for GPU acceleration.
Set --use_gpu to False if you want to use CPU instead.
Make sure also to set the ```device``` parameter in the .yaml config files situated in config/stgn_swmv/Dataset_name/train.yaml to the number of GPUs on your computer.
If you encountred memory errors try to reduce the ```batch_size```.

If you any questions or problems regarding the code, please contact us at : <mejdi.dallel@gmail.com> / <mdallel@cesi.fr>.

## Citation
To cite this work, please use:
``` 
@InProceedings{stgcn_swmv,  
author = {Mejdi Dallel, Vincent Havard, David Baudry and Yohan Dupuis},  
journal = {Conference on Information Sciences and Systems (CISS) 2022},
title = {A Sliding Window Based Approach With Double Majority Voting for Online Human Action Recognition using Spatial Temporal Graph Convolutional Neural Networks},   
year = {2022},    
pages = {1-6}
}
```
