from __future__ import absolute_import
import os
import sys
import tensorflow as tf
from Utils.UtilsFunctions import *
from STMModel.STMModelNet import *

from tensorflow.keras.layers import Layer

class STMModel(tf.keras.Model):
	def __str__(self):
		return str(self._stmModel)

	def __init__(self,
		F=None,                           #dimensionality of feature vector
		K=None,                           #number of radial basis functions
		depthtype=0, 			   # 0=> distance to surface, 1=> distances to atoms , 2=> 2 parameters : distance to plane and z of atoms, 3=> 4 parameters : distance to plane and x,yz, z of atoms
		cutoff=None,                      #cutoff distance for short range interactions
		num_hidden_nodes_em = None,       # number of nodes on each hidden layer in elemental modes block
		num_hidden_layers_em = 2,         #  number of hidden layers in elemental modes block
		num_blocks=5,                     #number of building blocks to be stacked
		num_residual_atomic=2,            #number of residual layers for atomic refinements of feature vector
		num_residual_interaction=3,       #number of residual layers for refinement of message vector
		num_residual_output=1,            #number of residual layers for the output blocks
		num_pixels=1024*1024,             #number of pixels
		negative_image=0,                 # >=0 no negative, >0 => negative
            drop_rate=None,                   #initial value for drop rate (None=No drop)
		initializer_name="GlorotNormal",  # initializer layer
		activation_fn=shifted_softplus,   #activation function
		output_activation_fn=tf.nn.relu,  # output activation function
		dtype=tf.float32,                 #single or double precision
		loss_type='SID',                  # loss type (SID, MAE)
		loss_bypixel=0,                   # loss by pixel if >0
		nhlambda=0,			  # lambda multiplier for non-hierarchicality loss (regularization)
		basis_type="Default",             # radial basis type : GaussianNet (Default for MPNN), Gaussian(Default for EANN), Bessel, Slater, 
		seed=None):
		super().__init__(dtype=dtype, name="STMModel")

		self._stmModel=None

		self._num_outputs=num_pixels
		self._negative_image=negative_image
		self._stmModel=STMModelNet(
						F=F,
						K=K,
						depthtype=depthtype,
						num_hidden_nodes_em =  num_hidden_nodes_em,
						num_hidden_layers_em = num_hidden_layers_em,
						cutoff=cutoff,
						dtype=dtype, 
						num_blocks=num_blocks, 
						num_residual_atomic=num_residual_atomic,
						num_residual_interaction=num_residual_interaction,
						num_residual_output=num_residual_output,
						num_outputs=self.num_outputs, 
						initializer_name=initializer_name,
						activation_fn=activation_fn,
						output_activation_fn=output_activation_fn,
						drop_rate=drop_rate,
						nhlambda=nhlambda,
						basis_type=basis_type,
						loss_type=loss_type,
						loss_bypixel=loss_bypixel,
						seed=seed)

	def computeProperties(self, data):
		return self._stmModel.computeProperties(data)

	def computeLoss(self, data):
		return self._stmModel.computeLoss(data)

	def print_parameters(self):
		self._stmModel.print_parameters()

	def __call__(self, data, closs=True):
		return self._stmModel(data,closs=closs)

	@property
	def stmModel(self):
		return self._stmModel

	@property
	def dtype(self):
		return self._stmModel.dtype

	@property
	def neuralNetwork(self):
		return self._stmModel.neuralNetwork

	@property
	def nhlambda(self):
		return self._stmModel.nhlambda

	@property
	def cutoff(self):
		return self._stmModel.cutoff

	@property
	def num_outputs(self):
		return self._num_outputs

	@property
	def negative_image(self):
		return self._negative_image
