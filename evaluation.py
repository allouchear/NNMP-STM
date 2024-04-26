import tensorflow as tf

import numpy as np
from Utils.Evaluator import *
from Utils.UtilsTrain import *
import os
import argparse
import shutil
import sys


def buildDatabase(data_dir,sys_ID_file, metrics_dir):
	if not os.path.exists(metrics_dir):
		os.makedirs(metrics_dir)
	new_data_dir=os.path.join(metrics_dir,  'database')
	if not os.path.exists(new_data_dir):
		os.makedirs(new_data_dir)
	outCSVIndex = os.path.join(new_data_dir,  'index.csv')

	print(sys_ID_file)
	dID = pd.read_csv(sys_ID_file)
	print(dID)
	IDToGet = dID['idxs'].to_numpy()
	print(IDToGet)
	data_file = os.path.join(data_dir,  'index.csv')
	df=pd.read_csv(data_file,index_col=None)
	print("Shape of original data file = ", df.shape)
	df = df.loc[IDToGet]
	print("Shape of new data file = ", df.shape)
	idxs=df["idxs"].to_numpy()
	fnames=df["fnames"].to_numpy()
	print(idxs)
	print(fnames)
	df.to_csv(outCSVIndex,index=False)
	for i, idx in enumerate(df.index):
		afile=os.path.join(data_dir,df['fnames'][idx])
		bfile=os.path.join(new_data_dir,df['fnames'][idx])
		shutil.copy2(afile,bfile)
	
	afile=os.path.join(data_dir,"parameters.csv")
	bfile=os.path.join(new_data_dir,"parameters.csv")
	print(afile, bfile)
	shutil.copy2(afile,bfile)

	return new_data_dir 

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--data_dir', default="Data/database", type=str, help="database directory. Default Data/database")
	parser.add_argument('--average', default=0, type=int, help="1 to Use average parameters instead best ones, 0 for best parameters")
	parser.add_argument('--list_models', type=str, nargs='+', help="list of directory containing fitted models (at least one file), ....")
	parser.add_argument('--metrics_dir', default="metrics_evaluation", type=str, help="directory where result are saved")
	parser.add_argument('--sys_ID_file', default="none", type=str, help="the name of file containing the ID cas of systems, default = None")
	parser.add_argument('--batch_size', default=32,type=int, help="batch size, default=32")
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

if "NONE" in args.sys_ID_file.upper():
	data_dir = args.data_dir
else:
	data_dir = buildDatabase(args.data_dir, args.sys_ID_file, args.metrics_dir)

lmodels=args.list_models
lmodels=lmodels[0].split(',')
metrics_dir=args.metrics_dir
batch_size=args.batch_size
print("Models = ", lmodels)
print("Data directory = ", data_dir)
print("Metrcis directory = ", metrics_dir)
print("Batch size = ", batch_size)
num_jobs=args.num_jobs

evaluator = Evaluator(
		lmodels,
		data_dir=data_dir,
		nvalues=-1,  # -1 for all values in datfile
		#nvalues=2, # 2 for test 
		batch_size=batch_size,
		average=args.average>0,
		num_jobs=num_jobs
		)

print("Accuraties for :", evaluator.nvalues, "values")
print("---------------------------------------------")
acc=evaluator.computeAccuracies(verbose=True)
print_all_acc(acc)
print("Save images in :", metrics_dir)
evaluator.saveImages(metrics_dir, dataType=0, uid=None)
evaluator.saveIndexes(metrics_dir, idxtype=0, uid=None)

