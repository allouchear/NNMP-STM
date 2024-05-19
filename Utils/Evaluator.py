import tensorflow as tf
import os
from Utils.DataContainer import *
from Utils.DataProvider import *
from STMModel.STMModel import *


def getExampleData(dtype):
	dtype=tf.float64 if dtype=='float64' else tf.float32
	data = {}
	data['Z'] =  [1,1]
	data['M'] = [1.0,1.0]
	data['R'] =  [[0.0,0.0,0.0],[1.0,0.0,0.0]]
	data['idx_i'] =  [0]
	data['idx_j'] =  [1]
	data['offsets'] = [0]
	data['sys_idx'] = tf.zeros_like(data['Z'])
	data['QaAlpha'] = tf.zeros_like(data['M'],dtype=dtype)
	data['QaBeta'] = tf.zeros_like(data['M'],dtype=dtype)
	data['dists']   = tf.zeros_like(data['M'],dtype=dtype)
	data['emins']   = tf.zeros_like(data['M'],dtype=dtype)
	data['emaxs']   = tf.zeros_like(data['M'],dtype=dtype)
	data['inputkeys']=["Z", 'dists', "emins", "emaxs", "M", "QaAlpha", "QaBeta"]
	return data

class Evaluator:
	def __init__(self, 
		models_directories,               # directories containing fitted models (can also be a list for ensembles)
		data_dir=os.path.join("Data","database"), 
		nvalues=-1,
		batch_size=1, 
		average=False, 
		num_jobs=1
		):


		if(type(models_directories) is not list):
			self._models_directories=[models_directories]
		else:
			self._models_directories=models_directories
		self._average = average
		self._data_dir = data_dir

		_, num_pixels, _, negative, dtype_p = readDatasetParameters(data_dir)

		self._data_negative_image = negative 

		self._data=DataContainer(data_dir, num_jobs)
		size=self.data.sys_idxs.shape[0]
		if nvalues<0:
			ntrain = size
		else:
			ntrain = nvalues
		self._nvalues=ntrain
		nvalid=0
		ntest=0
		valid_batch_size=0
		self._dataProvider=DataProvider(self.data,ntrain, nvalid, ntest=ntest, batch_size=batch_size,valid_batch_size=valid_batch_size)

		self._models = []
		n=0
		num_pixels=None
		loss_bypixel=None
		loss_type=None
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
			elif dtype_p != args.dtype:
				print("*************************************************************************************")
				print("Error type in data ", dtype_p, "!= ", args.dtype, "dtype of the model ", directory)
				print("*************************************************************************************")
				sys.exit(1)

			if loss_bypixel is None:
				loss_bypixel=args.loss_bypixel
			elif loss_bypixel != args.loss_bypixel:
				print("*************************************************************************************************")
				print("Warning : Model trained with different loss. We will use here loss_bypixel=",loss_bypixel)
				print("*************************************************************************************************")

			if loss_type is None:
				loss_type=args.loss_type
			elif loss_type != args.loss_type:
				print("*************************************************************************************************")
				print("Warning : Model trained with different loss. We will use here loss_type=",loss_type)
				print("*************************************************************************************************")


			data = getExampleData(args.dtype)
			images, loss, gradients = stmModel(data,closs=False) # to set auto shape
			#print("checkpoint=",checkpoint)
			stmModel.load_weights(checkpoint)
			self._models.append(stmModel)
		self._num_pixels  = num_pixels
		self._loss_bypixel  = loss_bypixel  
		self._loss_type  = loss_type

	def _computeProperties(self, data):
		n=0
		nModel=len(self._models)
		images = None
		nhloss =None
		for i in range(nModel):
			limages, lnhloss = self._models[i].computeProperties(data)
			if (self._models[i].negative_image>0) != (self.data_negative_image>0):
				limages = 1.0-limages

			if i == 0:
				images  = limages
				nhloss = lnhloss
			else:
				n = i+1
				if limages is not None: 
					images = images + (limages-images)/n
				if lnhloss is not None: 
					nhloss = nhloss+ (lnhloss-nhloss)/n 

		return images, nhloss

	def computeSums(self, data):
		images, nhloss = self._computeProperties(data)
		sums= {}
		values = { 'images':images, 'hloss':nhloss}
		coefs = { 'images':1.0, 'hloss':self.model.stmModel.nhlambda}
		for key in coefs:
			if coefs[key] > 0:
				datakey=tf.reshape(tf.constant(data[key],dtype=self.model.stmModel.dtype),[-1])
				predkey=tf.reshape(values[key],[-1])
				nan_mask=tf.math.logical_or(tf.math.is_nan(predkey) , tf.math.is_nan(datakey))
				datakey = tf.where(nan_mask, 0.0, datakey)
				predkey = tf.where(nan_mask, 0.0, predkey)
				sData=tf.reduce_sum(datakey)
				sPredict=tf.reduce_sum(predkey)
				s2Data=tf.reduce_sum(tf.square(datakey))
				s2Predict=tf.reduce_sum(tf.square(predkey))
				prod=datakey*predkey
				sDataPredict=tf.reduce_sum(prod)
				s2DataPredict=tf.reduce_sum(tf.square(prod))
				sdif=tf.math.abs(predkey-datakey)
				sAbsDataMPredict=tf.reduce_sum(sdif)
				"""
				sData=tf.reduce_sum(tf.reshape(tf.constant(data[key],dtype=self.model.stmModel.dtype),[-1]))
				sPredict=tf.reduce_sum(tf.reshape(values[key],[-1]))
				s2Data=tf.reduce_sum(tf.square(tf.reshape(tf.constant(data[key],dtype=self.model.stmModel.dtype),[-1])))
				s2Predict=tf.reduce_sum(tf.square(tf.reshape(values[key],[-1])))
				sDataPredict=tf.reduce_sum((tf.reshape(tf.constant(data[key],dtype=self.model.stmModel.dtype),[-1]))*tf.reshape(values[key],[-1]))
				s2DataPredict=tf.reduce_sum(tf.square((tf.reshape(tf.constant(data[key],dtype=self.model.stmModel.dtype),[-1]))*tf.reshape(values[key],[-1])))
				sAbsDataMPredict=tf.reduce_sum(tf.math.abs((tf.reshape(tf.constant(data[key],dtype=self.model.stmModel.dtype),[-1]))-tf.reshape(values[key],[-1])))
				"""
				d = {}
				d['sData']         =  sData
				d['sPredict']      =  sPredict
				d['s2Data']        =  s2Data
				d['s2Predict']     =  s2Predict
				d['sDataPredict']  =  sDataPredict
				d['s2DataPredict'] =  s2DataPredict
				d['sAbsDataMPredict'] = sAbsDataMPredict
				#d['n'] =  tf.reshape(values[key],[-1]).shape[0]
				d['n'] = datakey.shape[0]
				if key=='images':
					nx=data['image_nx']
					ny=data['image_ny']
					d['loss'] =  stm_loss(images, tf.constant(data[key],dtype=self.model.stmModel.dtype), nx, ny, loss_bypixel=self.loss_bypixel, loss_type=self.loss_type)
					if self.loss_bypixel<=0 and len(images.shape)>1:
						d['loss'] *=  len(images.shape) # because sums[key]=> sums[keys]['loss']/sums[keys]['n'] in computeAccuraciesFromSums
					else:
						d['loss'] *= d['n'] # because sums[key]=> sums[keys]['loss']/sums[keys]['n'] in computeAccuraciesFromSums
				elif key=='hloss':
					d['loss'] =  nhloss*d['n']
				else:
					d['loss'] =  sAbsDataMPredict
					#print("nOther=",d['n'])
				sums[key] = d

		return sums

	def addSums(self, data, sums=None):
		coefs = { 'images':1.0, 'hloss':self.model.stmModel.nhlambda}
		if sums is None:
			sums= {}
			lis=[ 'sData', 'sPredict', 's2Data', 's2Predict', 'sDataPredict', 's2DataPredict', 'sAbsDataMPredict','loss','n']
			for key in coefs:
				if coefs[key] > 0:
					d = {}
					for name in lis:
						d[name]  =  0
					sums[key] = d
		s=self.computeSums(data)
		for keys in sums:
			for key in sums[keys]:
				sums[keys][key] +=  s[keys][key]
		return sums

	def addTrainSums(self, sums=None):
		sums=self.addSums(self.dataProvider.current_batch(), sums=sums)
		return sums

	def computeAccuraciesFromSums(self, sums, verbose=False,dataType=0):
		coefs = { 'images':1.0, 'hloss':self.model.stmModel.nhlambda}
		acc = {}
		mae = {}
		ase = {}
		rmse = {}
		R2 = {}
		rr = {}
		lossV = {}
		for keys in sums:
			mae[keys] =  sums[keys]['sAbsDataMPredict']/sums[keys]['n']
			lossV[keys] =  sums[keys]['loss']/sums[keys]['n']
			m =  sums[keys]['sData']/sums[keys]['n']
			residual =  sums[keys]['s2Data'] +sums[keys]['s2Predict'] -2*sums[keys]['sDataPredict']
			if tf.math.abs(residual)<1e-14:
				R2[keys] = tf.Variable(1.0,dtype=sums[keys]['sData'].dtype)
			else:
				total =  sums[keys]['s2Data']-2*m*sums[keys]['sData']+sums[keys]['n']*m*m
				R2[keys] = 1.0-residual/(total+1e-14)

			ymean = m
			ypredmean = sums[keys]['sPredict']/sums[keys]['n']
			ase[keys] = ypredmean-ymean 
			rmse[keys] = tf.math.sqrt(tf.math.abs(residual/sums[keys]['n']))

			yvar =  sums[keys]['s2Data']-2*ymean*sums[keys]['sData']+sums[keys]['n']*ymean*ymean
			ypredvar =  sums[keys]['s2Predict']-2*ypredmean*sums[keys]['sPredict']+sums[keys]['n']*ypredmean*ypredmean
			cov =  sums[keys]['sDataPredict']-ypredmean*sums[keys]['sData']-ymean*sums[keys]['sPredict']+sums[keys]['n']*ymean*ypredmean
			den = tf.sqrt(yvar*ypredvar)
			if den<1e-14:
				corr = tf.Variable(1.0,dtype=sums[keys]['sData'].dtype)
			else:
				corr = cov/tf.sqrt(yvar*ypredvar+1e-14)
			rr[keys] = corr*corr

		loss=0.0
		for key in coefs:
			#if mae[key] in locals()  and coefs[key]>0:
			if coefs[key]>0:
				loss += lossV[key]*coefs[key]

		lossDic ={ 'L':loss}
		if self.loss_bypixel<=0:
			for key in mae.keys():
				mae[key] *= self.num_pixels
			for key in ase.keys():
				ase[key] *= self.num_pixels
			for key in rmse.keys():
				rmse[key] *= self.num_pixels
		acc['mae'] = mae
		acc['ase'] = ase
		acc['rmse'] = rmse
		acc['R2'] = R2
		acc['r2'] = rr
		acc['Loss'] = lossDic
		if verbose is True:
			if dataType==0:
				print("Metrics")
			elif dataType==1:
				print("Validation metrics")
			else:
				print("Test metrics")
			print("--------------------------------")
			for keys in acc:
				#print(keys,":")
				#[ print(keys,"[", key,"]=", acc[keys][key].numpy()) for key in acc[keys] ]
				[ print("{:5s}[{:2s}] = {:20.10f}".format(keys,key,acc[keys][key].numpy())) for key in acc[keys] ]
				#print("")
		return acc

	def computeLossFromSums(self, sums):
		coefs = { 'images':1.0}
		lossV = {}
		for keys in sums:
			lossV[keys] =  sums[keys]['loss']/sums[keys]['n']

		loss=0.0
		for key in coefs:
			if coefs[key]>0:
				loss += lossV[key]*coefs[key]

		return loss.numpy()

	def computeAccuracies(self, verbose=True, dataType=0):
		sums= None
		if dataType==0:
			nsteps = self.dataProvider.get_nsteps_batch()
		elif dataType==1:
			nsteps = self.dataProvider.get_nsteps_valid_batch()
		else:
			nsteps = self.dataProvider.get_nsteps_test_batch()

		if dataType==0:
			self.dataProvider.set_train_idx_to_end() # needed for train idx beaucuse we shuffle train part after each epoch

		for i in range(nsteps):
			if nsteps>10 :
				print("Compute accuracies ",i+1,"/",nsteps,flush=True)
			if dataType==0:
				dt = self.dataProvider.next_batch()
			elif dataType==1:
				dt = self.dataProvider.next_valid_batch()
			else:
				dt = self.dataProvider.next_test_batch()
		sums=self.addSums(dt, sums)
		if nsteps>0:
			acc = self.computeAccuraciesFromSums(sums, verbose,dataType=dataType)
		else: 
			acc= None
		return acc


	"""
		Save reference and predicted images in png
	"""
	def saveImagesBatch(self, data, pred, dirname):
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		
		for i in range(len(data['sys_ID'])):
			st="_"+str(data['sys_ID'][i])+".png"
			iref = data['images'][i]

			nx=data['image_nx'][i]
			ny=data['image_ny'][i]
			nxny=nx*ny
			iref= tf.reshape(iref[:nxny],[nx,ny,1])
			if self.data_negative_image>0:
				iref=1.0-iref # black <=> white
			fref=os.path.join(dirname,"ref"+st)
			saveOneImage(iref, fref)
			ipred = pred[i]
			if self.data_negative_image>0:
				ipred=1.0-ipred # black <=> white
			ipred= tf.reshape(ipred[:nxny],[nx,ny,1])
			fpred=os.path.join(dirname,"pred"+st)
			saveOneImage(ipred, fpred)

	def computeAndSaveImages(self, data, dirname):
		images, nhloss = self._computeProperties(data)
		if len(images.shape)==1:
			images=tf.reshape(images,[1,-1])
		sis=stm_sis(data['images'],images)
		nx=data['image_nx']
		ny=data['image_ny']
		ssim=stm_ssim_all(images, data['images'], nx,ny)
		ms_ssim=stm_ms_ssim_all(images, data['images'], nx,ny)
		mae=stm_mae_all(images, data['images'])
		rmse=stm_rmse_all(images, data['images'])
		df = pd.DataFrame({"SIS":sis.numpy(),"SSIM": ssim.numpy(), "MS_SSIM": ms_ssim.numpy(), "MAE":mae, "RMSE":rmse, "ID":data['sys_ID']})
		self.saveImagesBatch(data, images, dirname)
		return df

	def saveImages(self, metrics_dir, dataType=0, uid=None):
		if dataType==0:
			prefix=os.path.join(metrics_dir,"train")
		elif dataType==1 :
			prefix=os.path.join(metrics_dir,"validation")
		else:
			prefix=os.path.join(metrics_dir,"test")
		if uid is not None:
			prefix=prefix+"_"+str(uid)

		dirname = prefix+"_stm"

		if dataType==0:
			nsteps = self.dataProvider.get_nsteps_batch()
		elif dataType==1:
			nsteps = self.dataProvider.get_nsteps_valid_batch()
		else:
			nsteps = self.dataProvider.get_nsteps_test_batch()

		if dataType==0:
			self.dataProvider.set_train_idx_to_end() # needed for train idx beaucuse we shuffle train part after each epoch

		df = None
		for i in range(nsteps):
			if dataType==0:
				dt = self.dataProvider.next_batch()
			elif dataType==1:
				dt = self.dataProvider.next_valid_batch()
			else:
				dt = self.dataProvider.next_test_batch()
			#save_xyz(dt, "Train_"+id_generator()+".xyz")
			dfi=self.computeAndSaveImages(dt, dirname)
			if df is None:
				df = dfi
			else:
				df = pd.concat([df,dfi])

		if df is not None:
			fname=os.path.join(dirname,"similarities.csv")
			df.to_csv(fname,index=False)


		return dirname

	"""
		Save indexes in dataframe, one column
	"""
	def saveIdxs(self, idxs, dirname,idxtype=0):
		if not os.path.exists(dirname):
			os.makedirs(dirname)

		df=pd.DataFrame({"idxs":idxs})
		st='index_train.csv'
		if idxtype==1:
			st='index_validation.csv'
		elif idxtype==2:
			st='index_test.csv'

		fname=os.path.join(dirname,st)
		df.to_csv(fname,index=False)

	def saveIndexes(self, dirname, idxtype=0, uid=None):
		idxs = None
		if idxtype==0:
			idxs = self.dataProvider.idx_train
		elif idxtype==1:
			idxs = self.dataProvider.idx_valid
		else:
			idxs = self.dataProvider.idx_test
		if (idxs is not None) and  len(idxs>0):
			self.saveIdxs(idxs, dirname,idxtype=idxtype)

		return dirname



	@property
	def data(self):
		return self._data
	@property
	def dataProvider(self):
		return self._dataProvider

	@property
	def models(self):
		return self._models

	@property
	def model(self):
		return self._models[0]
    
	@property
	def nvalues(self):
		return self._nvalues

	@property
	def data_dir(self):
		return self._data_dir

	@property
	def data_negative_image(self):
		return self._data_negative_image

	@property
	def loss_type(self):
		return self._loss_type

	@property
	def loss_bypixel(self):
		return self._loss_bypixel

	@property
	def num_pixels(self):
		return self._num_pixels


