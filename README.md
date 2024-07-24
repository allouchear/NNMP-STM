# NNMP-STM  A neural network message passing model to predict STM image from 3D structure
=========================================================================================

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Requirement
 - tensorflow, 
 - tensorflow_probability, 
 - tensorflow_addons, 
 - nvidia-tensorrt

After installation of conda, and activation of your envoronnement,  Type : 
```console
pip install tensorflow==2.12.0
pip install tensorflow_probability==0.19.0
pip install tensorflow_addons==0.20.0
pip3 install nvidia-tensorrt
```

## Installation

Using git,  Type : 
```console
git clone https://github.com/allouchear/NNMP-STM.git
```
You can also download the .zip file of NNMP-STM : Click on Code and Download ZIP

## Main programs
### buildData.py. 
	see xbuildData script : from csv& png of Pierre => database for the training program

### train.py : using the database created by buildData.py, make the training. See traing.inp & xtrain
               ./xtrain to run in interactive. submitNNMP_CPU or submitNNMP_GPU(in2p3 only) to submit the job
		type submitNNMP_CPU or submitNNMP_GPU without parameters to show all options

### evaluation.py : test the models (one or anensemble of models) using a database
               ./xevaluation to run in interactive. submitNNMP_CPU or submitNNMP_GPU(in2p3 only) to submit the job
### predict.py : predict the STM image using a POSCAR file
               ./xpredict to run in interactive. submitNNMP_CPU or submitNNMP_GPU(in2p3 only) to submit the job


## Scripts (bash)
xbuildData   => build data in interactive 
xevaluationEnsemble => make an evalution.
xpredict	=> make an prediction. 
xtrain	=> make an train.

## Example
see example directory

## Contributors
The code is written by Abdul-Rahman Allouche.
