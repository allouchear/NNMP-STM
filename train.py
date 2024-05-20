import tensorflow as tf
import tensorflow_addons as tfa

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

import numpy as np
from Utils.DataContainer import *
from Utils.DataProvider import *
from STMModel.STMModel import *
from Utils.UtilsFunctions import *
from Utils.Trainer import *
from Utils.UtilsTrain import *
import os
import sys
import psutil


args = getArguments()


directory, log_dir, best_dir, average_dir,  metrics_dir, best_checkpoint, average_checkpoint, step_checkpoint, best_loss_file =setOutputLocationFiles(args)


#print("activation_fn=",activation_deserialize(args.activation_function))
print("activation_fn=",activation_deserialize(args.activation_function).__name__)
#print("activation_fn=",activation_deserialize(args.activation_function))
print("output activation_fn=",activation_deserialize(args.output_activation_function).__name__)

#********************* Creation of NN Model *************** 
logging.info("Creation of NN model")
stmModel=   STMModel (F=args.num_features,
			K=args.num_basis,
			cutoff=args.cutoff,
			dtype=tf.float64 if args.dtype=='float64' else tf.float32, 
			num_hidden_nodes_em=args.num_hidden_nodes_em,
			num_hidden_layers_em=args.num_hidden_layers_em,
			num_pixels=args.num_pixels,
			num_blocks=args.num_blocks, 
			num_residual_atomic=args.num_residual_atomic,
			num_residual_interaction=args.num_residual_interaction,
			num_residual_output=args.num_residual_output,
			initializer_name=args.initializer,
			activation_fn=activation_deserialize(args.activation_function),
			output_activation_fn=activation_deserialize(args.output_activation_function),
			drop_rate=args.drop_rate,
			nhlambda=args.nhlambda,
			basis_type=args.basis_type,
			loss_type=args.loss_type,
			loss_bypixel=args.loss_bypixel,
			seed=args.seed)


print("============================ Model =================================================================",flush=True)
print(stmModel)
print("====================================================================================================",flush=True)

nsteps=args.max_steps

#********************* Creation of trainer model *************** 
logging.info("Creation of trainer")
trainer = create_trainer(stmModel, args)

st = "Save indexes lists in "+metrics_dir+" directory"
logging.info(st)
trainer.saveIndexes(metrics_dir,idxtype=0)
trainer.saveIndexes(metrics_dir,idxtype=1)
trainer.saveIndexes(metrics_dir,idxtype=2)

save_model_parameters(args)

logging.info("Creation of histograms for data")
trainer.dataProvider.create_tensorboard_histograms(log_dir)

logging.info("Load best recorded loss")
lossbest, props = load_best_recorded_loss(best_loss_file)

if args.load_average>0:
	logging.info("Trying to load of average check point")
	print("Trying to load of average check point")
	ok=trainer.load_weights(fname=average_checkpoint)
else:
	logging.info("Trying to load of best check point")
	print("Trying to load of best check point")
	ok=trainer.load_weights(fname=best_checkpoint)



#********************* Open metrics files *************** 
validation_metrics_files, validation_metrics_file_names = open_validation_metrics_files(metrics_dir,  args)
train_metrics_files, train_metrics_file_names = open_train_metrics_files(metrics_dir,  args)
test_metrics_files, test_metrics_file_names = open_test_metrics_files(metrics_dir,  args)

nstepsByEpoch = trainer.dataProvider.get_nsteps_batch()

#********************* Training *************** 
logging.info("Begin training")
print("Begin training",flush=True)
sums=None

#===================== create log dirs =============
writer_logs_train = tf.summary.create_file_writer(os.path.join(log_dir,  'train'))
writer_logs_validation = tf.summary.create_file_writer(os.path.join(log_dir,  'validation'))
writer_logs_test = tf.summary.create_file_writer(os.path.join(log_dir,  'test'))

#===================== Training loop =============== 
nmaxtr=100
nmaxte=100
for istep in range(nsteps):
	epoch=istep//nstepsByEpoch
	print("Step : ", istep+1,"/",nsteps, "; Epoch = ", epoch+1)
	print("------------------------------",flush=True)
	#dt=trainer.dataProvider.next_batch()
	if args.verbose>=1:
		print("Current learning rate = ",trainer.get_learning_rate())
	#print("dt.E=",dt['E'])
	#loss,gradients= trainer.applyOneStep(dt=dt, verbose=False)
	loss,gradients= trainer.applyOneStep(verbose=False)

	if args.verbose>=1:
		print_gradients_norms(gradients,stmModel.trainable_weights,details=args.verbose>=2)
	#print_gradients_norms(gradients,stmModel.trainable_weights,details=False)

	#sums=trainer.addTrainSums(sums=sums)
	#aloss=trainer.computeLossFromSums(sums)
	#print("Step : ", istep+1,"/",nsteps, "; Epoch = ", epoch+1, " ; Loss=",  loss.numpy(), " ; Averaged Loss=", aloss) 
	#print("Step : ", istep+1,"/",nsteps, "; Epoch = ", epoch+1, " ; Loss=",  loss.numpy())
	print("Loss=",  loss.numpy(),flush=True)

	#if istep%args.validation_interval==1000:
	if (istep+1)%args.validation_interval==0:
		lossbest=validation_test(trainer, lossbest, best_checkpoint, best_loss_file, istep)
		if trainer.use_average==1:
			trainer.save_variable_backups()
			trainer.set_average_vars()
			trainer.save_weights(fname=average_checkpoint)
			trainer.restore_variable_backups()

	if (istep+1)%args.save_interval==0:
		if trainer.use_average==1:
			trainer.save_variable_backups()
			trainer.set_average_vars()
		print("Saving images .....",flush=True)
		dirname=trainer.saveImages(metrics_dir,dataType=1,uid=(istep+1))
		if trainer.dataProvider.ntrain<=nmaxtr:
			trainer.saveImages(metrics_dir,dataType=0,uid=(istep+1)) 
		if trainer.dataProvider.ntest<=nmaxte:
			trainer.saveImages(metrics_dir,dataType=2,uid=(istep+1)) 
		if trainer.use_average==1:
			trainer.restore_variable_backups()

	if (istep+1)%args.summary_interval==0:
		print("Saving logs .....",flush=True)
		acc=add_validation_metrics_to_files(trainer, validation_metrics_files, istep==0)
		add_metrics_to_logs(writer_logs_validation, acc, istep, prefix=None)

		if trainer.dataProvider.ntrain<=nmaxtr:
			acc=add_train_metrics_to_files(trainer, train_metrics_files, istep==0) # can be very long
			add_metrics_to_logs(writer_logs_train, acc, istep, prefix=None)

		if trainer.dataProvider.ntest<=nmaxte:
			acc=add_test_metrics_to_files(trainer, test_metrics_files, istep==0)
			add_metrics_to_logs(writer_logs_test, acc, istep, prefix=None)
	nc=100; [ print("=",end='') if i<nc-1 else print("=",flush=True) for i in range(nc) ]

#exit(1)
#********************* Testing of fitted variables on training & validation data  *************** 
lossbest=validation_test(trainer, lossbest, best_checkpoint, best_loss_file, nsteps-1)
print("====================================================================================================",flush=True)
logging.info("Begin test on all training data & all validation ones")

trainer.save_weights(fname=step_checkpoint)
# Update the weights to their mean before saving
print("Averaged variables")
print("==================")
trainer.save_variable_backups()
if trainer.use_average==1:
	trainer.set_average_vars()

print("Train data",flush=True)
print("----------")
acc=trainer.computeTrainAccuracies(verbose=False)
print_all_acc(acc)
print("Saving images .....",flush=True)
dirname=trainer.saveImages(metrics_dir,dataType=0, uid='final')

print("Validation data",flush=True)
print("--------------")
acc=trainer.computeValidationAccuracies(verbose=True)
print_all_acc(acc)
print("Saving images .....",flush=True)
dirname=trainer.saveImages(metrics_dir,dataType=1, uid='final')

print("Test data",flush=True)
print("--------------")
acc=trainer.computeTestAccuracies(verbose=False)
print_all_acc(acc)
print("Saving images .....",flush=True)
dirname=trainer.saveImages(metrics_dir,dataType=2, uid='final_best')

trainer.save_weights(fname=average_checkpoint)
print("----------------------------------------------------------")

#********************* Testing of best variables on training, validation, test data  *************** 
print("Best variables",flush=True)
#trainer.restore_variable_backups()
trainer.load_weights(fname=best_checkpoint)
print("==============")
print("Train data",flush=True)
print("----------")
acc=trainer.computeTrainAccuracies(verbose=False)
print_all_acc(acc)
print("Save training data metrics in :", metrics_dir)
print("Saving images .....",flush=True)
dirname=trainer.saveImages(metrics_dir,dataType=0,uid='final_best')

print("Validation data",flush=True)
print("--------------")
acc=trainer.computeValidationAccuracies(verbose=False)
print_all_acc(acc)
print("Save validation data metrics in :", metrics_dir)
print("Saving images .....",flush=True)
dirname=trainer.saveImages(metrics_dir,dataType=1, uid='final_best')

print("Test data",flush=True)
print("--------------")
print("Save test data metrics in :", metrics_dir)
acc=trainer.computeTestAccuracies(verbose=False)
print_all_acc(acc)
print("Saving images .....",flush=True)
dirname=trainer.saveImages(metrics_dir,dataType=2, uid='final_best')

print("====================================================================================================",flush=True)
logging.info("That'a all. Good bye")
print("That'a all. Good bye")
close_metrics_files(validation_metrics_files)
print("See metrics in files ")
print("---------------------")
for key in validation_metrics_file_names:
	print(validation_metrics_file_names[key])
for key in train_metrics_file_names:
	print(train_metrics_file_names[key])

print("Logs files for tendorboard ")
print("---------------------")
print("logs files are in ", log_dir)
print("To visualise them : tensorboard --logdir ", log_dir)
print(" and run your navigator to show all ")
print("====================================================================================================",flush=True)
