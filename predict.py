import tensorflow as tf

import numpy as np
from Utils.Predictor import *
from Utils.UtilsFunctions import *
import os
import argparse
import shutil
import sys
from ase import io


def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--list_models', type=str, nargs='+', help="list of directory containing fitted models (at least one file), ....")
	parser.add_argument('--average', default=0, type=int, help="1 to Use average parameters instead best ones, 0 for best parameters")
	parser.add_argument('--input_file_name', default="POSCAR", type=str, help="POSCAR input file")
	parser.add_argument('--output_file_name', default="POSCAR.png", type=str, help="PNG input file")
	parser.add_argument("--conv_distance", type=float, default=1.0, help=" convertion factor for distance. Default 1.0 (same unit in data and fitted model)")
	parser.add_argument("--conv_energy", type=float, default=1.0, help=" convertion factor for distance. Default 1.0 (same unit in data and fitted model)")
	parser.add_argument("--conv_mass", type=float, default=1.0, help=" convertion factor for distance. Default 1.0 (same unit in data and fitted model)")
	parser.add_argument('--distance', default=1.0, type=float, help="dist as used in VASP. Default 1.0")
	parser.add_argument('--emin', default=-10.0, type=float, help="EMIN as used in VASP. Default -10.0")
	parser.add_argument('--emax', default=0.0, type=float, help="EMAX as used in VASP. Default 0.0")
	parser.add_argument('--num_jobs', default=1,type=int, help="number of jobs for // calculations. Default=1")

	#if no command line arguments are present, config file is parsed
	config_file='config.txt'
	fromFile=False
	if len(sys.argv) == 1:
		fromFile=False
	if len(sys.argv) == 2 and sys.argv[1].find('--') == -1:
		config_file=sys.argv[1]
		fromFile=True

	if fromFile is True:
		print("Try to read configuration from ",config_file, "file")
		if os.path.isfile(config_file):
			args = parser.parse_args(["@"+config_file])
		else:
			args = parser.parse_args(["--help"])
	else:
		args = parser.parse_args()

	return args

args = getArguments()
lmodels=args.list_models
lmodels=lmodels[0].split(',')
print("---------------------------------------------")
print("Models = ", lmodels)
num_jobs=args.num_jobs

atoms=io.read(args.input_file_name)
predictor = Predictor(
		lmodels,
		atoms,
		distance=args.distance,   
		emin=args.emin,   
		emax=args.emax,   
		conv_distance=args.conv_distance,
		conv_energy=args.conv_energy,
		conv_mass=args.conv_mass,
		average=args.average>0,
		num_jobs=num_jobs
		)

image=predictor.computeSTM() # return real image
print("Image shape ", image.shape)
#print("image", image)
saveOneImage(image, args.output_file_name)
print("See ", args.output_file_name, " file ")
print("---------------------------------------------")
