import argparse
import pandas as pd
import numpy as np
import sys
import os
import tensorflow as tf
import ase
from ase import io
from joblib import Parallel, delayed
from joblib import cpu_count
import h5py
from ase.neighborlist import neighbor_list
from Utils.PeriodicTable import *
from Utils.UtilsFunctions import *

periodicTable=PeriodicTable()

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--infile", type=str, default="data.csv", help="data.csv file.")
	parser.add_argument("--outdir", type=str, default="database", help="processed data output directory")
	parser.add_argument("--seed", type=int, default=111, help=" seed, default 111. If <0 => random_state=None")
	parser.add_argument("--p", type=float, default=100.0, help=" real : % of selected structures, default=100")
	parser.add_argument("--njobs", type=int, default=-1, help=" number of jobs. All procs if <0. Default -1")
	parser.add_argument("--conv_distance", type=float, default=1.0, help=" convertion factor for distances. Default 1.0")
	parser.add_argument("--conv_energy", type=float, default=1.0, help=" convertion factor for energy. Default 1.0")
	parser.add_argument("--conv_mass", type=float, default=1.0, help=" convertion factor for mass. Default 1.0")
	parser.add_argument("--cutoff", type=float, default=10.0, help=" cutoff distance in Bohr. Default 10.0")
	parser.add_argument("--printimages", type=int, default=0, help=" printimages : 0=> No, 1=> yes. Default 0")
	parser.add_argument("--num_pixels", type=int, default=1024*1024, help=" number of pixels (integer) :. Default 1048576 (1024*1024)")
	parser.add_argument("--dtype", type=str, default="float32", help=" type of float : float32 or float64 :. Default float32")
	parser.add_argument("--negative_image", type=int, default=0, help="Data in negative (>0) or not(<=0): Default 0=>no negative")
	args = parser.parse_args()
	return args

def get_idx_list(atoms, cutoff=None):
		if any(atoms.get_pbc()):
			if cutoff is None:
				print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
				print("cutoff  is needed for periodic system")
				print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
				sys.exit(1)

		if cutoff is not None:
			srcal=False
			idx_i, idx_j, S = neighbor_list('ijS', atoms, cutoff, self_interaction=False)
			offsets = np.dot(S, atoms.get_cell())
		else:
			idx_i = []
			idx_j = []
			for ia in range(N):
				for ja in range(N):
					if ja != ia:
						idx_i.append(ia)
						idx_j.append(ja)
			offsets=[None]*len(idx_i)
		return idx_i, idx_j, offsets


def getData(args):
	dtype=args.dtype
	filepath=args.infile
	df = pd.read_csv(filepath, sep=',')
	filter =  df['struct']=='struct'
	df = df[~filter]
	df['Directory'] = df['struct'].apply(os.path.dirname)
	df['PNG'] = df['Directory'] + os.sep + df['pngfile']
	df["emin"]=df["Emin"].to_numpy(dtype=dtype)*args.conv_energy
	df["emax"]=df["Emax"].to_numpy(dtype=dtype)*args.conv_energy
	df["dist"]=df["dist"].to_numpy(dtype=dtype)*args.conv_distance
	df = df[['struct', 'PNG', 'dist',  'emin', 'emax']]
	#df=df.rename(columns={"Emin": "emin", "Emax": "emax"})
	df.reset_index(inplace=True,drop=True)
	#print(df)
	#print(df['struct'][0])
	#print(df['PNG'][0])
	seed = args.seed if args.seed>=0 else None
	if(args.p<=100):
		print("Shuffle data")
		df = df.sample(frac = args.p/100, replace = False, random_state = seed)
		df.reset_index(drop = True, inplace=True)
		# print the shuffled dataframe to easily find problematic structures if any
		df.to_csv(args.outdir + os.sep + "data_train.csv", index=True)
		#print(df)
		#print("Number of selected data = ",df.shape[0])

	return df

def printAtoms(atoms):
	print("--------- Input geometry --------------------")
	print("PBC : ",atoms.get_pbc())
	print("Cell : " ,atoms.get_cell())
	print("Z : " , atoms.get_atomic_numbers())
	print("Positions : ",atoms.get_positions())
	print("---------------------------------------------")

def getMasses(Z):
	masses = []
	for za in Z:
		if int(za)==0:
			masses.append(0)
		else:
			mass=periodicTable.elementZ(int(za)).isotopes[0].rMass # mass of first istotope
			masses.append(mass)
	return masses

def getQa(Z):
	QaAlpha = [0]*len(Z)
	QaBeta  = [0]*len(Z)
	return QaAlpha,QaBeta

def getOneStructure(directory, idx, fname,conv_distance,conv_mass, cutoff):
	atoms = io.read(fname)
	idx_i, idx_j, offsets = get_idx_list(atoms, cutoff/conv_distance) # atoms are in Ansstrom
	pbc = atoms.get_pbc()
	cell = atoms.get_cell()
	Z=atoms.get_atomic_numbers()
	positions=atoms.get_positions()
	masses = getMasses(Z)
	QaAlpha, QaBeta = getQa(Z)
	cell *= conv_distance
	positions *= conv_distance
	offsets *= conv_distance
	masses = np.array(masses)*conv_mass

	bn=os.path.basename(fname)
	ofname=bn+"_"+str(idx)
	ofname= os.path.join(directory, ofname)
	st="cp " + fname +" "+ofname
	os.system(st)
	#print("see structure ", fname ," copied in ", ofname)

	return pbc, cell, Z, positions, masses, QaAlpha, QaBeta, idx_i, idx_j, offsets

def getNxNy(cell,nxny=1024*1024):
	a=tf.norm(cell[0])
	b=tf.norm(cell[1])
	c=tf.norm(cell[2])
	ba=b/a
	nx=tf.math.sqrt(nxny*ba)
	ny=tf.math.sqrt(nxny/ba)
	nx=tf.cast(nx, dtype=tf.uint32)
	ny=tf.cast(ny, dtype=tf.uint32)
	return nx,ny

def getOneImage(directory,idx,fname,cell,printimages=False,num_pixels=1024*1024,prefix="sys_",dtype='float32', negative=False):
	image = tf.io.read_file(fname)
	#image = tf.io.decode_png(image, channels = 1,dtype=tf.uint8) # 1 => output a grayscale image.
	image = tf.io.decode_png(image, channels = 1,dtype=tf.uint16) # 1 => output a grayscale image.
	#print("max v image int ",tf.math.reduce_max(image))
	dtype=tf.float64 if dtype=='float64' else tf.float32
	image=tf.image.convert_image_dtype(image, dtype=dtype)
	if negative:
		image=1.0-image # invert black-white
	#print("max v image float ",tf.math.reduce_max(image))

	nx,ny= getNxNy(cell,nxny=num_pixels)
	#image = tf.image.resize(image, [nx,ny],antialias=False) # return float32
	#image = tf.image.resize(image, [nx,ny],antialias=True) # return float32
	image = tf.image.resize(image, [nx,ny],antialias=False, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # return float64
	if printimages:
		if negative:
			sfname=prefix+str(idx)+"_scaled_negative.png"
			sfname= os.path.join(directory, sfname)
			saveOneImage(image, sfname)

		sfname=prefix+str(idx)+"_scaled.png"
		sfname= os.path.join(directory, sfname)
		if negative:
			saveOneImage(1-image, sfname)
		else:
			saveOneImage(image, sfname)

		ofname=prefix+str(idx)+"_original.png"
		ofname= os.path.join(directory, ofname)
		st="cp " + fname +" "+ofname
		os.system(st)
		#print("see rescaled image = ", sfname ," and original one ", ofname)
	#print(image.shape)
	return image

def buildOnSystem(directory, df,idx, conv_distance, conv_mass, cutoff, prefix="sys_", printimages=False, num_pixels=1024*1024, dtype='float32',negative=False):
	fname=prefix+str(idx)+".h5"
	outfile= os.path.join(directory, fname)
	store = h5py.File(outfile,'w')
	pbc, cell, Z, positions , masses ,QaAlpha, QaBeta, idx_i, idx_j, offsets = getOneStructure(directory, idx, df['struct'][idx], conv_distance, conv_mass, cutoff)
	image = getOneImage(directory, idx, df['PNG'][idx],cell, printimages=printimages,num_pixels=num_pixels, prefix=prefix,dtype=dtype,negative=negative)
	store["sys_ID"]=[idx]
	store["pbc"]=pbc
	store["cell"]=cell
	store["R"]=positions
	store["Z"]=Z
	store["M"]=masses
	store["QaAlpha"]=QaAlpha
	store["QaBeta"]=QaBeta
	store["idx_i"]=idx_i
	store["idx_j"]=idx_j
	store["offsets"]=offsets
	nr=num_pixels-image.shape[0]*image.shape[1]
	if nr<0:
		printf("********************************************")
		printf("Error : image size > num_pixels")
		printf("********************************************")
		sys.exit()
	rest=tf.constant([np.nan]*nr,dtype=dtype)
	#print("rest size = ",rest.shape)
	image1D=tf.concat([tf.reshape(image,-1), rest],axis=-1)
	#print("image 1D size = ",image1D.shape)
	store["image"]= image1D
	store["image_nx"]= [image.shape[0]]
	store["image_ny"]= [image.shape[1]]
	store["dist"]= [df["dist"][idx]]
	store["emin"]= [df["emin"][idx]]
	store["emax"]= [df["emax"][idx]]
	store.close()
	return fname, image.shape

def buildData(directory,df,conv_distance, conv_mass, cutoff,prefix="sys_", printimages=False,num_pixels=1024*1024, dtype='float32',negative=False):
	lidx=df.shape[0]
	fnames=[]
	i=1
	for idx in df.index:
		fname,shape = buildOnSystem(directory, df,idx, conv_distance, conv_mass, cutoff, prefix=prefix,printimages=printimages,num_pixels=num_pixels,dtype=dtype, negative=negative)
		fnames.append(fname)
		print('% : {:5.2f} ; {:5d}/{:5d} size = {}\r'.format((i)/lidx*100, i, lidx,shape),end='')
		i += 1
	print()
	return fnames

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

def buildAllData(directory, df, conv_distance, conv_mass, cutoff, njobs=1, prefix="sys_",printimages=False,num_pixels=1024*1024, dtype='float32',negative=False):
	print("Data processing ... ")
	print("Number of structures =",df.shape[0])
	if njobs==1:
		fnames = buildData(directory,df, conv_distance, conv_mass, cutoff, prefix=prefix,printimages=printimages,num_pixels=num_pixels,dtype=dtype, negative=negative)
		return fnames
	else:
		NB,NE = getNBNE(njobs, df.shape[0])
		r = Parallel(n_jobs=njobs, verbose=20)(
			delayed(buildData)(directory,df[NB[i]:NE[i]], conv_distance, conv_mass, cutoff, prefix=prefix,printimages=printimages,num_pixels=num_pixels,dtype=dtype,negative=negative) for i in range(njobs)
		) 
		fnames = [element for innerList in r for element in innerList]
	#print(fnames)
	return fnames


def getdirname(args ) :
def # print the shuffled dataframe to easily find problematic structures if any
def directory=args.outdir
	if not os.path.exists(directory):
		os.makedirs(directory)
	return directory

def getnjobs(args):
	njobs = cpu_count()
	#if args.njobs >0 and args.njobs<njobs:
	if args.njobs >0:
		njobs=args.njobs
	return njobs

def buildIndexFile(df, directory, fnames):
	print(df.columns)
	idxs=df.index.to_numpy(dtype=np.uint32)
	poscar=df.struct.to_numpy()
	png=df.PNG.to_numpy()
	'''
	outdata= os.path.join(directory, 'index.h5')
	store = h5py.File(outdata,'w')
	store["fnames"]=fnames
	store["idxs"]=idxs
	store.close()
	'''
	dic = {'idxs': idxs, 'fnames': fnames, 'poscar':poscar, 'png':png} 
	dfi=pd.DataFrame(dic)
	outindexcsv= os.path.join(directory, 'index.csv')
	dfi.to_csv(outindexcsv,index=False)

def buildParametersFile(directory, cutoff, num_pixels,prefix="sys_",negative=False,dtype='float32'):
	ng=[1 if negative else 0]
	if ng[0]>=1:
		print("==========================\nNegative images\n===========================")
	else:
		print("==========================\nPositive images\n===========================")
	dic = {'cutoff': [cutoff],'num_pixels':[num_pixels], "prefix":[prefix],"negative":ng,"dtype":dtype} 
	dfi=pd.DataFrame(dic)
	outcsv= os.path.join(directory, 'parameters.csv')
	dfi.to_csv(outcsv,index=False)

def readDataOneSystem(directory, idx, prefix="sys_"):
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

def readDataSystems(directory, idxs, prefix="sys_"):
	data =[]
	for idx in idxs:
		d = readDataOneSystem(directory, idx, prefix=prefix)
		data += [d]
	return data

# To test reading by index list
def readDataAllSystems(directory, idxs, njobs=1, prefix="sys_"):
	print("Data reading ... for idx=",idxs)
	print("Number of structures =",len(idxs))
	datalist =[]
	if njobs==1:
		datalist = readDataSystems(directory, idxs, prefix=prefix)
	else:
		NB,NE = getNBNE(njobs, len(idxs))
		r = Parallel(n_jobs=njobs, verbose=20)(
			delayed(readDataSystems)(directory,idxs[NB[i]:NE[i]], prefix=prefix) for i in range(njobs)
		) 
		if len(r)>0:
			for d in r:
				if len(d)>0:
					datalist.extend(d)

	data = {}
	if len(datalist)<1:
		return data
	#print(datalist[0][0])
	keys=datalist[0].keys()
	Ntot=0
	data['sys_idx'] = []
	for i,d in enumerate(datalist):
		for key in keys:
			dd=d[key]
			if key=="idx_i" or key=="idx_j":
				dd += Ntot

			if key in data.keys():
				data[key] = np.concatenate((data[key], dd))
			else:
				data[key] = dd
		data['sys_idx'].extend([i] * data["N"][i])
		print("i=",i," idx=", "N=",data["N"][i])
		Ntot += data["N"][i]
	data['sys_idx'] = np.array(data["sys_idx"])
	data['image'] = tf.reshape(data['image'], [len(datalist),-1])
	data['images'] = data.pop('image')
		
	for key in data.keys():
		print("Key = ", key, " Shape =", data[key].shape, "Type = ", data[key].dtype)
	'''
	with np.printoptions(threshold=np.inf):
		print(data["Z"])
		print(data["R"])
		print(data["idx_i"])
	'''
	return data

def testReadData(directory, df, njobs=1, seed=0, prefix="sys_"):
	random_state = np.random.RandomState(seed=seed)
	lend=df.shape[0]
	idx = random_state.permutation(np.arange(lend))
	n=int(0.2*lend)
	if n<1:
		n=1
	idxs=df.index[0:n]
	data = readDataAllSystems(directory, idxs, njobs=njobs, prefix=prefix)


print("========================================================================================================================")
args = getArguments()
df=getData(args)
njobs = getnjobs(args)
print("Number of process=",njobs,"/",cpu_count())
directory = getdirname(args)
outcsv= os.path.join(directory, 'out.csv')
df.to_csv(outcsv)
prefix="sys_"
negative=negative=args.negative_image>0
fnames =buildAllData(directory, df, args.conv_distance, args.conv_mass, args.cutoff, njobs=njobs, prefix=prefix,printimages=args.printimages,num_pixels=args.num_pixels,dtype=args.dtype, negative=negative)
buildIndexFile(df, directory, fnames)
buildParametersFile(directory, args.cutoff, args.num_pixels,prefix=prefix, negative=negative,dtype=args.dtype)

print("========================================================================================================================")
print("Number of structures =",df.shape[0])
print("See files in ", directory, " directory")
print("========================================================================================================================")
#print("test reading a part of files by indexes ")
#testReadData(directory, df, seed=args.seed, njobs=njobs, prefix=prefix)
#print("========================================================================================================================")


