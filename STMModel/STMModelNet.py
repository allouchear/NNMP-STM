from __future__ import absolute_import
import os
import tensorflow as tf
from Utils.ActivationFunctions import *
from Utils.UtilsFunctions import *
from Utils.UtilsLoss import *
from .NeuralNetwork import *

from tensorflow.keras.layers import Layer

class STMModelNet(tf.keras.Model):
	def __str__(self):
		st = str(self.neuralNetwork)
		st +="\n"\
            + "Loss type                                              : "\
		+ self.loss_type\
		+"\n"
		return st

	def __init__(self,
		F=128,                            #dimensionality of feature vector
		K=64,                             #number of radial basis functions
		depthtype=0, 			    # 0=> distance to surface, 1=> distances to atoms , 2=> 2 parameters : distance to plane and z of atoms")
		cutoff=8.0,                       #cutoff distance for short range interactions (atomic unit)
		num_hidden_nodes_em = None,       # number of nodes on each hidden layer in elemental modes block
		num_hidden_layers_em = 2,         #  number of hidden layers in elemental modes block
		num_blocks=5,                     #number of building blocks to be stacked
		num_residual_atomic=2,            #number of residual layers for atomic refinements of feature vector
		num_residual_interaction=3,       #number of residual layers for refinement of message vector
		num_residual_output=1,            #number of residual layers for the output blocks
		num_outputs=2,        		    #number of outputs by atom
            drop_rate=None,                   #initial value for drop rate (None=No drop)
		initializer_name="GlorotNormal",    # nitializer layer
		activation_fn=shifted_softplus,   #activation function
		output_activation_fn=tf.nn.relu,  # output activation function
		dtype=tf.float32,                 #single or double precision
		loss_type='SID',                  # loss type (SID, MAE)
		loss_bypixel=0,                   # loss by pixel if >0
		nhlambda=0,			  # lambda multiplier for non-hierarchicality loss (regularization)
		basis_type="Default",             # radial basis type : GaussianNet (Default for MPNN), Gaussian(Default for EANN), Bessel, Slater, 
		seed=None):
		super().__init__(dtype=dtype, name="STMModelNet")
		self._loss_type = loss_type
		self._loss_bypixel = loss_bypixel

		self._neuralNetwork = neuralNetwork(
			F=F,
			K=K,
			depthtype=depthtype,
			num_hidden_nodes_em = num_hidden_nodes_em,
			num_hidden_layers_em = num_hidden_layers_em,
			basis_type=basis_type,
			cutoff=cutoff, 
			num_blocks=num_blocks, 
			num_residual_atomic=num_residual_atomic, 
			num_residual_interaction=num_residual_interaction,
			num_residual_output=num_residual_output, 
			num_outputs=num_outputs,
			drop_rate=drop_rate,
			initializer_name=initializer_name,
			activation_fn=activation_fn,
			output_activation_fn=output_activation_fn,
			dtype=dtype,
			seed=seed) 


		self._nhlambda=nhlambda

		self._dtype=dtype


	def computeProperties(self, data):
		outputs, nhloss = self.neuralNetwork.atomic_properties(data)
		images  = outputs
		images = tf.squeeze(tf.math.segment_mean(images, data['sys_idx']))
		#images = tf.squeeze(tf.math.segment_sum(images, data['sys_idx']))

		return images, nhloss

	def computeLoss(self, data):
		with tf.GradientTape() as tape:
			images, nhloss = self.computeProperties(data)
			nx=data['image_nx']
			ny=data['image_ny']
			loss = stm_loss(images, tf.constant(data['images'],dtype=self.dtype), nx, ny, loss_bypixel=self.loss_bypixel, loss_type=self.loss_type)
			#loss /= tf.reshape(images,[-1]).shape[0] DEBUG
			#print("lossDiv=",loss)
			if self.nhlambda>0:
				loss += self.nhlambda*nhloss

		gradients = tape.gradient(loss, self.trainable_weights)
		#print("Loss=",loss)
		#print("mean=",mean)
		#print("-------Loss-----------------\n",loss,"\n----------------------------\n")
		#print("-------Gradients------------\n",gradients,"\n----------------------------\n")
		return images, loss, gradients

	def __call__(self, data, closs=True):
		if closs is not True:
			images, nhloss = self.computeProperties(data)
			loss=None
			gradients=None
		else:
			images, loss, gradients = self.computeLoss(data)
			
		return images, loss, gradients

	def print_parameters(self):
		pass

	@property
	def dtype(self):
		return self._dtype

	@property
	def neuralNetwork(self):
		return self._neuralNetwork

	@property
	def nhlambda(self):
		return self._nhlambda

	@property
	def loss_type(self):
		return self._loss_type

	@property
	def loss_bypixel(self):
		return self._loss_bypixel

	@property
	def cutoff(self):
		return self.neuralNetwork.cutoff


