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
### buildData.py : Build data for training.
see xbuildData script in example directory.

### train.py
**Using the database created by buildData.py, make the training**
see train.inp and ./xtrain bash script in example directory.

### evaluation.py
**Test the models (one or anensemble of models) using a database**
see  ./xevaluationEnsemble bash script in example directory.

### predict.py
**Predict the STM image using a POSCAR file**
see ./xpredict bash script in example directory.


## Example
see example directory

## Contributors
The code is written by Abdul-Rahman Allouche.
