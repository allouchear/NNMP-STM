import tensorflow as tf
import numpy as np
import ase
from STMModel.STMModel import *
from Utils.UtilsFunctions import *
from Utils.PhysicalConstants import *
from ase.neighborlist import neighbor_list
from Utils.PeriodicTable import *

periodicTable=PeriodicTable()

def getExampleData():
	data = {}
	data['Z'] =  [1,1]
	data['M'] = [1.0,1.0]
	data['R'] =  [[0.0,0.0,0.0],[1.0,0.0,0.0]]
	data['idx_i'] =  [0]
	data['idx_j'] =  [1]
	data['offsets'] = [0]
	data['sys_idx'] = tf.zeros_like(data['Z'])
	data['QaAlpha'] = tf.zeros_like(data['M'])
	data['QaBeta'] = tf.zeros_like(data['M'])
	data['dists']   = tf.zeros_like(data['M'])
	data['emins']   = tf.zeros_like(data['M'])
	data['emaxs']   = tf.zeros_like(data['M'])
	data['inputkeys']=["Z", 'dists', "emins", "emaxs", "M", "QaAlpha", "QaBeta"]
	return data

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

def getOneStructure(atoms, conv_distance,conv_mass,cutoff):
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

	return pbc, cell, Z, positions, masses, QaAlpha, QaBeta, idx_i, idx_j, offsets

def getData(atoms, conv_distance, conv_energy, conv_mass, cutoff, dist, emin, emax):
	pbc, cell, Z, positions , masses ,QaAlpha, QaBeta, idx_i, idx_j, offsets = getOneStructure(atoms,conv_distance,conv_mass, cutoff)
	data={}
	data["sys_ID"]=[0]
	data["pbc"]=pbc
	data["cell"]=cell
	data["R"]=positions
	data["Z"]=Z
	data["M"]=masses
	data["QaAlpha"]=QaAlpha
	data["QaBeta"]=QaBeta
	data["idx_i"]=idx_i
	data["idx_j"]=idx_j
	data["offsets"]=offsets
	N = len(Z)
	data["dists"]= [dist*conv_distance]*N
	data["emins"]= [emin*conv_energy]*N
	data["emaxs"]= [emax*conv_energy]*N
	data["sys_idx"]= [0]*N
	data['inputkeys']=["Z", 'dists', "emins", "emaxs", "M", "QaAlpha", "QaBeta"]
	return data


	
class Predictor:
	def __init__(self,
		models_directories,               # directories containing fitted models (can also be a list for ensembles)
		atoms,                            #ASE atoms object
		distance=1,                       # Distance as used with VAPS
		emin=-10,                         # Emin as used in VASP
		emax=0,                           # Emax as used in VASP
		conv_distance=1.0,        #coef. conversion of distance
		conv_energy=1.0,         #coef. conversion of energies 
		conv_mass=1.0,                    #coef. conversion of mass
		average=False, 
		num_jobs=1
		):

		if(type(models_directories) is not list):
			self._models_directories=[models_directories]
		else:
			self._models_directories=models_directories
		self._average = average
		self._atoms = atoms

		self._conv_distance = conv_distance
		self._conv_energy = conv_energy
		self._conv_mass = conv_mass

		self._distance = distance
		self._emin = emin
		self._emax = emax

		self._models = []
		n=0
		num_pixels=None
		for directory in self._models_directories:
			args = read_model_parameters(directory)
			if self._average:
				average_dir = os.path.join(directory, 'average')
				checkpoint = os.path.join(average_dir, 'average.ckpt')
			else:
				best_dir = os.path.join(directory, 'best')
				checkpoint = os.path.join(best_dir, 'best.ckpt')

			self._cutoff=args.cutoff
			stmModel=   STMModel (F=args.num_features,
			K=args.num_basis,
			cutoff=args.cutoff,
			dtype=tf.float64 if args.dtype=='float64' else tf.float32, 
			num_hidden_nodes_em=args.num_hidden_nodes_em,
			num_hidden_layers_em=args.num_hidden_layers_em,
			num_blocks=args.num_blocks, 
			num_residual_atomic=args.num_residual_atomic,
			num_residual_interaction=args.num_residual_interaction,
			num_residual_output=args.num_residual_output,
			num_pixels=args.num_pixels,
			negative_image=args.negative_image,
			activation_fn=activation_deserialize(args.activation_function),
			output_activation_fn=activation_deserialize(args.output_activation_function),
			basis_type=args.basis_type,
			loss_type=args.loss_type,
			)
			if num_pixels is None:
				num_pixels=args.num_pixels
			elif num_pixels != args.num_pixels:
				print("**********************************************************")
				print("Error number of pixels are not the same in all models")
				print("**********************************************************")
				sys.exit(1)
			

			data = getExampleData()
			images, loss, gradients = stmModel(data,closs=False) # to set auto shape
			#print("checkpoint=",checkpoint)
			stmModel.load_weights(checkpoint)
			self._models.append(stmModel)
		self._num_pixels  = num_pixels

	def _computeProperties(self):
		data = None
		n=0
		nModel=len(self._models)
		atoms= self._atoms
		for i in range(nModel):
			if i==0 or self._models[i].cutoff != self._models[i-1].cutoff:
				self._data = getData(atoms, self.conv_distance, self.conv_energy, self.conv_mass, self.cutoff, self.distance, self.emin, self.emax)
			image, nhloss = self._models[i].computeProperties(self.data)
			if self._models[i].negative_image>0:
				image = 1.0-image # real image not negative one, so we return real image. Several models can be fitted with or without negatives images
			#print("atomcharges=",atomcharges)

			if i == 0:
				self._image  = image
			else:
				n = i+1
				if image is not None:
					self._image +=  (image-self._image)/n 


		self._image = self._image.numpy() 

	def computeSTM(self):
		self._computeProperties()
		nx,ny=self.image_shape()
		nxny=nx*ny
		im =self._image[:nxny]
		#im /=  tf.math.reduce_max(im)
		image= tf.reshape(im,[nx,ny,1])
		return image

	@property
	def atoms(self):
		return self._atoms

	@property
	def image(self):
		return self._image

	@property
	def model(self):
        	return self._models

	@property
	def checkpoint(self):
		return self._checkpoint

	@property
	def conv_distance(self):
		return self._conv_distance

	@property
	def conv_energy(self):
		return self._conv_energy

	@property
	def conv_mass(self):
		return self._conv_mass

	@property
	def num_pixels(self):
		return self._num_pixels

	@property
	def cutoff(self):
		return self._cutoff

	@property
	def distance(self):
		return self._distance

	@property
	def emin(self):
		return self._emin

	@property
	def emax(self):
		return self._emax

	@property
	def data(self):
		return self._data

	def image_shape(self):
		cell = self.atoms.get_cell()
		a=tf.norm(cell[0])
		b=tf.norm(cell[1])
		ba=b/a
		nx=tf.math.sqrt(self.num_pixels*ba)+0.5
		nx=tf.cast(nx, dtype=tf.uint32)
		ny=self.num_pixels//nx
		ny=tf.cast(ny, dtype=tf.uint32)
		return nx,ny

