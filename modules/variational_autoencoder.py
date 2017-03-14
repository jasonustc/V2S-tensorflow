import tensorflow as tf

def xavier_init(fan_in, fan_out, constant = 1):
	low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
	high = constant * np.sqrt(6.0 / (fan_in + fan_out))
	return tf.random_uniform((fan_in, fan_out), min_val = low, max_val = high, dtype = tf.float32)

class VariationalAutoEncoder(object):
	"""
		Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
	"""

	def __init__(self, n_input, n_output):
		self.mu_W = tf.Variable(xavier_init(n_input, n_output), name = 'ae_rbm')
		self.mu_bias = tf.Variable(tf.zeros([n_output]), name = 'mu_bias')
		self.sigma_W = tf.Variable(xavier_init(n_input, n_output), name = 'ae_rbm')
		self.sigma_bias = tf.Variable(tf.zeros([n_output]), name = 'sigma_bias')

	def build_model(self, x, batch_size):
		z_mean = tf.nn.xw_plus_b(x, self.mu_W, self.mu_bias)
		z_log_sigma_sq = tf.nn.xw_plus_b(x, self.sigma_W, self.sigma_bias)
		z_sigma_sq = tf.exp(z_log_sigma_sq)
		eps = tf.random_normal(tf.shape(z_mean), 0., 1.)
		z = tf.add(z_mean, tf.multiply(tf.sqrt(z_sigma_sq)), eps)
		latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - z_sigma_sq, 1)
		return z, latent_loss
