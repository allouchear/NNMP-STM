import tensorflow as tf

def get_initializer(initialize, name,seed=None):
	if seed==None:
		initializer=tf.keras.initializers.deserialize(name)
		return initializer
	lists= ["GlorotNormal" ,"GlorotUniform","HeNormal" ,"HeUniform","LecunNormal" ,"LecunUniform", "RandomNormal", "RandomUniform", "runcatedNormal", "VarianceScaling"]
	if name in lists:
		config={'seed':seed}
		d={'class_name':name, 'config':config}
		initializer=tf.keras.initializers.deserialize(d)
		return initializer
	else:
		initializer=tf.keras.initializers.deserialize(name)
		return initializer
