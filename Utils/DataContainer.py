"""
Data from h5 using DataFrame
"""

import sys
import numpy  as np
import pandas as pd
from .PeriodicTable import *
from .UtilsFunctions import *
import h5py
import os
import psutil
import sys
from joblib import Parallel, delayed
import tensorflow as tf

def readIndexFile(directory):
	indexfile=os.path.join(directory, "index.csv")
	df = pd.read_csv(indexfile,index_col=None)
	return df

def readDataOneSystem(directory, idx, prefix):
	fname=prefix+str(idx)+".h5"
	outfile= os.path.join(directory, fname)
	store = h5py.File(outfile,'r')
	data={}
	for key in store.keys():
		data[key]= store[key][()]
	store.close()
	N=data["Z"].shape[0]
	data['N']=np.array([N])
	return data

def readDataSystems(directory, idxs, prefix):
	data =[]
	for idx in idxs:
		d = readDataOneSystem(directory, idx, prefix=prefix)
		data.append(d)
	return data

def getNBNE(njobs, dfshape):
	m =  dfshape//njobs
	M =[m]*njobs
	nr=dfshape%njobs
	for i in range(nr):
		M[i] += 1
	NB =[0]*njobs
	NE =[0]*njobs
	NE[0] = M[0]
	for i in range(1,njobs):
		NB[i] = NE[i-1]
		NE[i] = NB[i]+M[i]
	'''
	print("NB=",NB,flush=True)
	print("NE=",NE,flush=True)
	'''
	return NB, NE


# To test reading by index list
def readDataAllSystems(directory, idxs, njobs, prefix):
	#print("DEBUG Data reading ... for systems with index=",idxs)
	#print("DEBUG Number of structures =",len(idxs))
	datalist =[]
	if njobs==1:
		datalist = readDataSystems(directory, idxs, prefix)
	else:
		NB,NE = getNBNE(njobs, len(idxs))
		r = Parallel(n_jobs=njobs, verbose=20)(
			delayed(readDataSystems)(directory,idxs[NB[i]:NE[i]], prefix) for i in range(njobs)
		) 
		if len(r)>0:
			for d in r:
				if len(d)>0:
					datalist.extend(d)

	data = {}
	if len(datalist)<1:
		return data
	keys=datalist[0].keys()
	Ntot=0
	data['sys_idx'] = []
	data['dists'] = []
	data['emins'] = []
	data['emaxs'] = []
	for i in range(len(datalist)):
		d=datalist[i]
		for key in keys:
			dd=d[key]
			if key=="idx_i" or key=="idx_j":
				dd += Ntot

			if key in data.keys():
				data[key] = np.concatenate((data[key], dd))
			else:
				data[key] = dd
		data['sys_idx'].extend([i] * data["N"][i])
		data['dists'].extend([data['dist'][0]] * data["N"][i])
		data['emins'].extend([data['emin'][0]] * data["N"][i])
		data['emaxs'].extend([data['emax'][0]] * data["N"][i])
		#print("i=",i," idx=", "N=",data["N"][i])
		Ntot += data["N"][i]

	data['sys_idx'] = np.array(data["sys_idx"])
	data['image'] = tf.reshape(data['image'], [len(datalist),-1])
	data['images'] = data.pop('image')
	data['dists'] = np.array(data["dists"])
	data['emins'] = np.array(data["emins"])
	data['emaxs'] = np.array(data["emaxs"])
	del data['dist']
	del data['emin']
	del data['emax']
	data['inputkeys']=["Z", 'dists', "emins", "emaxs", "M", "QaAlpha", "QaBeta"]
	#print("DEBUG Data shape ",data['images'].shape)
	#print("DEBUG Meme percent : ",  psutil.Process(os.getpid()).memory_percent())
		
		
	'''
	for key in data.keys():
		print("Key = ", key, " Shape =", data[key].shape, "Type = ", data[key].dtype)
	with np.printoptions(threshold=np.inf):
		print(data["Z"])
		print(data["R"])
		print(data["idx_i"])
	'''
	return data

def getPrefix(directory):
	fname=os.path.join(directory, "parameters.csv")
	df=pd.read_csv(fname)
	return df["prefix"].to_numpy()[0]

class DataContainer:
	def __repr__(self):
		return "DataContainer"
	def __init__(self, directory, num_jobs):
		df=readIndexFile(directory)
		#print(df.index)
		print("=================== Database info ==================================================================")
		print("Number of structures=",df.shape[0])
		#print("Data info\n",df.info())
		print("columns=",df.columns.values)

		self._directory = directory
		self._num_jobs = num_jobs
		self._sys_idxs = df["idxs"].to_numpy()
		self._sys_fnames = df["fnames"].to_numpy()
		self._prefix = getPrefix(directory)
		print("sys_idxs=",self.sys_idxs)
		print("====================================================================================================")

	def __len__(self): 
		return self.sys_idxs.shape[0]

	@property
	def prefix(self): 
		return self._prefix

	@property
	def num_jobs(self):
		return self._num_jobs

	@property
	def directory(self):
		return self._directory

	@property
	def sys_idxs(self):
		return self._sys_idxs

	@property
	def sys_fnames(self):
		return self._sys_fnames

	def __getitem__(self, idxs): # idxs = index of sys_idxs and sys_fnames
		sys_idxs = self.sys_idxs[idxs]
		if type(idxs) is int or type(idxs) is np.int64:
			sys_idxs = [sys_idxs]
		data = readDataAllSystems(self.directory, sys_idxs, njobs=self.num_jobs, prefix=self.prefix)

		return data

