import tensorflow as tf
from tensorflow.keras.layers import Layer
from .RBFLayer import *
from .InteractionBlock import *
from .OutputBlock      import *
from .ElementalModesBlock import *
from Utils.ActivationFunctions import *
import psutil
import os

def softplus_inverse(x):
	'''numerically stable inverse of softplus transform'''
	return x + np.log(-np.expm1(-x))

class ElementalModesMessagePassingNeuralNetwork(Layer):
	def __str__(self):
		st =    "Elemental Modes Message Passing Neural Network\n"+\
			"----------------------------------------------"\
			+"\n"\
			+"Dimensionality of feature vector                       : " + str(self.F)\
			+"\n"+str(self.rbf_layer)\
			+"\n"\
			+"\n"\
			+"Number of hidden layers in element modes block         : " + str(len(self.elemental_modes_block.hidden_layers)) \
			+"\n"\
			+"Number of hidden nodes by layer in element modes block : " + str(self.elemental_modes_block.hidden_layers[0].get_config()['units']) \
			+"\n"\
			+"Number of building blocks                              : " + str(self.num_blocks)\
			+"\n"\
			+"Number of residuals for interaction                    : " + str(len(self.interaction_block[0].interaction.residual_layer))\
			+"\n"\
			+"Number of residuals for atomic                         : " + str(len(self.interaction_block[0].residual_layer))\
			+"\n"\
			+"Number of outputs                                      : " + str(self.num_outputs)\
			+"\n"\
			+"Number of residuals for outputs                        : " + str(len(self.output_block[0].residual_layer))\
			+"\n"\
			+"Initializer                                            : " + str(self.initializer_name)\
			+"\n"\
			+"Float type                                             : " + str(self.dtype)\

		if self.activation_fn is None:
			st = "\nActivation function                                     : None"
		else:
			st += "\nActivation function                                    : "+str(self.activation_fn.__name__)
		if self.output_activation_fn is None:
			st = "\nOutput activation function                              : None"
		else:
			st += "\nOutput activation function                             : "+str(self.output_activation_fn.__name__)

		return st

	def __init__(self,
		F,                               #dimensionality of feature vector
		K,                               #number of radial basis functions
		cutoff,                          #cutoff distance for short range interactions
		num_hidden_nodes_em = None, 	 #number of hidden nodes by layer in element modes block , None => F
		num_hidden_layers_em = 2, 	 #number of hidden layer in element modes block
		num_blocks=5,                    #number of building blocks to be stacked
		num_residual_atomic=2,           #number of residual layers for atomic refinements of feature vector
		num_residual_interaction=3,      #number of residual layers for refinement of message vector
		num_residual_output=1,           #number of residual layers for the output blocks
		num_outputs=2001,      		 #number of outputs by atom
            drop_rate=None,                  #initial value for drop rate (None=No drop)
		activation_fn=shifted_softplus,  #activation function
		initializer_name="GlorotNormal",    # nitializer layer
		output_activation_fn=tf.nn.relu, # output activation function
		basis_type="Default",            #radial basis type : GaussianNet (Default), Gaussian, Bessel, Slater, 
		beta=0.2,			 #for Gaussian basis type
		dtype=tf.float32,                #single or double precision
		seed=None):
		super().__init__(dtype=dtype, name="ElementalModesMessagePassingNeuralNetwork")

		assert(num_blocks > 0)
		assert(num_outputs > 0)
		self._num_blocks = num_blocks
		self._dtype = dtype

		self._num_outputs = num_outputs 

		self._F = F
		self._K = K
		self._cutoff = tf.constant(cutoff,dtype=dtype) #cutoff for neural network interactions
		
		self._activation_fn = activation_fn
		self._initializer_name = initializer_name
		self._output_activation_fn = output_activation_fn

		#drop rate regularization
		"""
		if drop_rate is None:
			self._drop_rate = tf.Variable(0.0, shape=[], name="drop_rate",dtype=dtype,trainable=False)
		else:
			self._drop_rate = tf.Variable(0.0, shape=[], name="drop_rate",dtype=dtype,trainable=True)
		"""
		self._drop_rate = tf.Variable(0.0, shape=[], name="drop_rate",dtype=dtype,trainable=False)


		#elemental_modes_block blocks and output layers
		self._elemental_modes_block = ElementalModesBlock(F, initializer_name=initializer_name, num_hidden_nodes=num_hidden_nodes_em, num_hidden_layers=num_hidden_layers_em, activation_fn=activation_fn, seed=seed, drop_rate=drop_rate, dtype=dtype, name="elemental_modes_block")

		#radial basis function expansion layer
		self._rbf_layer = RBFLayer(K,  self._cutoff, beta=beta, basis_type=basis_type, name="rbf_layer",dtype=dtype)

		#embedding blocks and output layers
		self._interaction_block = []
		self._output_block = []
		nouts = num_outputs 

		for i in range(num_blocks):
			self.interaction_block.append(
			InteractionBlock(F, num_residual_atomic, num_residual_interaction, initializer_name=initializer_name, activation_fn=activation_fn, name="InteractionBlock"+str(i),
					seed=seed, drop_rate=self.drop_rate, dtype=dtype))
			self.output_block.append(
				OutputBlock(F, nouts, num_residual_output, initializer_name=initializer_name, activation_fn=activation_fn, name="OutputBlock"+str(i),
					seed=seed, drop_rate=self.drop_rate, dtype=dtype))

	def calculate_interatomic_distances(self, R, idx_i, idx_j, offsets=None):
		#calculate interatomic distances
		Ri = tf.gather(R, idx_i)
		Rj = tf.gather(R, idx_j)
		if offsets is not None:
			Rj += offsets
		Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri-Rj)**2, -1))) #relu prevents negative numbers in sqrt
		return Dij

	def get_input_elements(self, data):

		f = None
		for key in data['inputkeys']:
			v =tf.Variable(data[key],dtype=self.dtype)
			v = tf.reshape(v,[v.shape[0],1])
			if f is None:
				f=v
			else:
				f=tf.concat([f,v],1)
		return f

	#calculates the atomic properties and distances (needed if unscaled charges are wanted e.g. for loss function)
	def atomic_properties(self, data):
		#calculate distances (for long range interaction)
		R=tf.Variable(data['R'],dtype=self.dtype)
		idx_i =  data['idx_i']
		idx_j =  data['idx_j']
		offsets=data['offsets']
		Dij = self.calculate_interatomic_distances(R, idx_i, idx_j, offsets=offsets)

		#calculate radial basis function expansion
		rbf = self.rbf_layer(Dij)
		#print("rbf=\n",rbf,"\n-------------------------------------\n")

		f = self.get_input_elements(data)
		x = self.elemental_modes_block(f)
		outputs = 0
		nhloss = 0 #non-hierarchicality loss
		for i in range(self.num_blocks):
			x = self.interaction_block[i](x, rbf, idx_i, idx_j)
			out = self.output_block[i](x)
			outputs += out
			#compute non-hierarchicality loss
			out2 = out[:]**2
			if i > 0:
				nhloss += tf.reduce_mean(out2/(out2 + lastout2 + 1e-7))
			lastout2 = out2

		outputs=self.output_activation_fn(outputs)
		return outputs, nhloss

	@property
	def drop_rate(self):
		return self._drop_rate

	@property
	def num_scc(self):
		return self._num_scc
    
	@property
	def num_blocks(self):
		return self._num_blocks

	@property
	def num_outputs(self):
		return self._num_outputs

	@property
	def dtype(self):
		return self._dtype

	@property
	def elemental_modes_block(self):
		return self._elemental_modes_block

	@property
	def F(self):
		return self._F

	@property
	def K(self):
		return self._K

	@property
	def em_type(self):
		return self._em_type

	@property
	def cutoff(self):
		return self._cutoff

	@property
	def activation_fn(self):
		return self._activation_fn

	@property
	def initializer_name(self):
		return self._initializer_name
    
    
	@property
	def output_activation_fn(self):
		return self._output_activation_fn
    
	@property
	def rbf_layer(self):
		return self._rbf_layer

	@property
	def interaction_block(self):
		return self._interaction_block

	@property
	def output_block(self):
		return self._output_block

