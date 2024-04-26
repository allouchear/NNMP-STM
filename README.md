# NNMP-STM

## Scripts (bash)
### scriptsLocal all bash scripts for a local machine (specially if you have a GPU).
### scriptsLynx all bash scripts for the cluster lynx.
### scriptsIN2P3 all bash scripts for the IN2P2 calculation center.

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

## Example : see example directory

## Warnning : If you change the directory of the code, go to scripts* directory and type ./xchdirCode
