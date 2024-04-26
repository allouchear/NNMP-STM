#!/bin/bash

cd /sto2/ssd/allouche/MySoftwares/NNMP-STM-11-Code/example
source /home/theochem/allouche/.bashrc

grep -v num_jobs train.inp > mod_train.inp
echo "--num_jobs=20" >> mod_train.inp

source /sto2/ssd/allouche/MySoftwares/NNMP-STM-11-Code/env.sh
export CUDA_VISIBLE_DEVICES='' # to force CPU use
time python -u /sto2/ssd/allouche/MySoftwares/NNMP-STM-11-Code/train.py mod_train.inp > train.out
/bin/rm train.sh
/bin/rm mod_train.inp
