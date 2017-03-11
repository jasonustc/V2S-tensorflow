import tensorflow as tf

def xavier_init(fan_in, fan_out, constant = 1):
	low = -constant * np.sqrt(6.0 / (fan_in + fan_out))

class VariationalAutoEncoder(object):
	"""
		Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
	"""

	def __init__(self, n_input, n_output):
		self.W = tf.Variable(tf.zeros([n_input, n_output]), name = 'ae_rbm')