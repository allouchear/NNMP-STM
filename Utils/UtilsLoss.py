#from __future__ import absolute_import
import tensorflow as tf
import numpy  as np
import sys


THRFREQ=1e-3

def stm_ssim(model_stm ,target_stm, nx, ny):
	model_stm_image  = tf.reshape(model_stm,[model_stm.shape[0],nx,ny,1])
	target_stm_image = tf.reshape(target_stm, [target_stm.shape[0],nx,ny,1])
	max_val = 1.0
	minm=tf.reduce_min(model_stm_image)
	maxm=tf.reduce_max(model_stm_image)
	model_stm_image = (model_stm_image-minm)/(maxm-minm)
	# A tensor containing an SSIM value for each image in batch or a tensor containing an SSIM value for each pixel for each image in batch 
	# if return_index_map is True. Returned SSIM values are in range (-1, 1], when pixel values are non-negative. 
	# Returns a tensor with shape: broadcast(img1.shape[:-3], img2.shape[:-3]) or broadcast(img1.shape[:-1], img2.shape[:-1]). 
	ssim = tf.image.ssim(model_stm_image, target_stm_image, max_val=max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, return_index_map=True)
	return ssim

def stm_ssim_loss(model_stm ,target_stm, nx, ny):
	ssim = stm_ssim(model_stm, target_stm, nx, ny)
	loss = 1-ssim
	return loss

def stm_ms_ssim (model_stm ,target_stm, nx, ny):
	model_stm_image  = tf.reshape(model_stm,[model_stm.shape[0],nx,ny,1])
	target_stm_image = tf.reshape(target_stm, [target_stm.shape[0],nx,ny,1])
	max_val = 1.0
	minm=tf.reduce_min(model_stm_image)
	maxm=tf.reduce_max(model_stm_image)
	model_stm_image = (model_stm_image-minm)/(maxm-minm)
	# Return A tensor containing an MS-SSIM value for each image in batch. The values are in range [0, 1]. 
      # Returns a tensor with shape: broadcast(img1.shape[:-3], img2.shape[:-3]). 
	ssim_ms = tf.image.ssim_multiscale(model_stm_image, target_stm_image, max_val=max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

	return ssim_ms

def stm_ms_ssim_loss(model_stm ,target_stm, nx, ny):
	ms_ssim = stm_ms_ssim(model_stm, target_stm, nx, ny)
	loss = 1-ms_ssim
	return loss

def stm_sid_loss(model_stm ,target_stm):
	loss = tf.math.log(model_stm/target_stm)*model_stm+\
		tf.math.log(target_stm/model_stm)*target_stm
	return loss

def stm_mae_loss(model_stm ,target_stm):
	loss = tf.abs(model_stm-target_stm)
	return loss

def stm_rmse_loss(model_stm ,target_stm):
	loss = (model_stm-target_stm)*(model_stm-target_stm)
	return loss

def stm_loss(model_stm ,target_stm, nx, ny, threshold=THRFREQ, loss_bypixel=0, loss_type='SID'):
	if len( model_stm.shape)==1:
		 model_stm = tf.reshape(model_stm,[1,-1])
		 target_stm = tf.reshape(target_stm,[1,-1])
	nan_mask=tf.math.logical_or(tf.math.is_nan(target_stm) , tf.math.is_nan(model_stm))

	if loss_type.upper()=='SID':
		model_stm = tf.where(model_stm<threshold, threshold, model_stm)
		target_stm = tf.where(target_stm<threshold, threshold, target_stm)
		model_stm = tf.where(nan_mask, 1.0, model_stm)
		target_stm = tf.where(nan_mask, 1.0, target_stm)
		loss = stm_sid_loss(model_stm ,target_stm)
		loss = tf.where(nan_mask, 0.0, loss)
		if loss_bypixel>0:
			loss=tf.reduce_mean(loss,axis=1)
		else:
			loss=tf.reduce_sum(loss,axis=1)
	elif loss_type.upper()=='RMSE':
		model_stm = tf.where(nan_mask, 0.0, model_stm)
		target_stm = tf.where(nan_mask, 0.0, target_stm)
		loss= stm_rmse_loss(model_stm ,target_stm)
		loss = tf.where(nan_mask, 0.0, loss)
		if loss_bypixel>0:
			loss=tf.reduce_mean(loss,axis=1) # by pixels
		else:
			loss=tf.reduce_sum(loss,axis=1)
		loss=tf.math.sqrt(loss)
	elif loss_type.upper()=='MAE':
		model_stm = tf.where(nan_mask, 0.0, model_stm)
		target_stm = tf.where(nan_mask, 0.0, target_stm)
		loss= stm_mae_loss(model_stm ,target_stm)
		loss = tf.where(nan_mask, 0.0, loss)
		if loss_bypixel>0:
			loss=tf.reduce_mean(loss,axis=1) # by pixels
		else:
			loss=tf.reduce_sum(loss,axis=1)
	elif loss_type.upper()=='SSIM':
		n = target_stm.shape[0]
		target_stm = tf.reshape(target_stm[~nan_mask],[n,-1])
		model_stm = model_stm[:,0:target_stm.shape[1]]
		loss= stm_ssim_loss(model_stm ,target_stm, nx, ny)
		if loss_bypixel>0:
			loss=tf.reduce_mean(loss,axis=1) # by pixels
		else:
			loss=tf.reduce_sum(loss,axis=1)
	elif loss_type.upper()=='MS_SSIM':
		n = target_stm.shape[0]
		target_stm = tf.reshape(target_stm[~nan_mask],[n,-1])
		model_stm = model_stm[:,0:target_stm.shape[1]]
		loss= stm_ms_ssim_loss(model_stm ,target_stm, nx, ny)
		# return a loss for each image
		if loss_bypixel<=0:
			loss *= nx*ny
	else:
		print("*************************************************")
		print("ERROR : Unknown loss type ",loss_type)
		print("        Calculation stopped")
		print("        Use SID, MAE , RMSE , SSIM or MS_SSIM")
		print("*************************************************")
		sys.exit()

	loss=tf.reduce_mean(loss) # loss = mean of loss structures. loss of a struture = sum of difference on all pixels if loss=tf.reduce_sum(loss,axis=1) , by pixels if tf.reduce_mean
	#loss=tf.reduce_sum(loss)
	return loss

def stm_sis(model_stm ,target_stm,threshold=THRFREQ):
	nan_mask=tf.math.logical_or(tf.math.is_nan(target_stm) , tf.math.is_nan(model_stm))
	model_stm = tf.where(model_stm<threshold, threshold, model_stm)
	target_stm = tf.where(target_stm<threshold, threshold, target_stm)
	model_stm = tf.where(nan_mask, 1.0, model_stm)
	target_stm = tf.where(nan_mask, 1.0, target_stm)

	loss = stm_sid_loss(model_stm ,target_stm)
	loss=tf.reduce_sum(loss,axis=1)
	npixels=target_stm[0].shape[0]
	#print("npixels=",npixels)
	loss /= npixels
	sis = 1.0/(1.0+loss)
	return sis

def stm_ssim_all(model_stm ,target_stm, nx, ny):
	if len( model_stm.shape)==1:
		 model_stm = tf.reshape(model_stm,[1,-1])
		 target_stm = tf.reshape(target_stm,[1,-1])
	nan_mask=tf.math.logical_or(tf.math.is_nan(target_stm) , tf.math.is_nan(model_stm))
	n = target_stm.shape[0]
	target_stm = tf.reshape(target_stm[~nan_mask],[n,-1])
	model_stm = model_stm[:,0:target_stm.shape[1]]
	ssim= stm_ssim(model_stm ,target_stm, nx, ny)
	ssim = tf.reshape(ssim,[ssim.shape[0],-1])
	ssim=tf.reduce_mean(ssim,axis=1) # by pixels
	return ssim

def stm_ms_ssim_all(model_stm ,target_stm, nx, ny):
	if len( model_stm.shape)==1:
		 model_stm = tf.reshape(model_stm,[1,-1])
		 target_stm = tf.reshape(target_stm,[1,-1])
	nan_mask=tf.math.logical_or(tf.math.is_nan(target_stm) , tf.math.is_nan(model_stm))
	n = target_stm.shape[0]
	target_stm = tf.reshape(target_stm[~nan_mask],[n,-1])
	model_stm = model_stm[:,0:target_stm.shape[1]]
	ms_ssim= stm_ms_ssim(model_stm ,target_stm, nx, ny)
	return ms_ssim
