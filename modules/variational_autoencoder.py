import tensorflow as tf
import numpy as np
import pdb

def xavier_init(fan_in, fan_out, constant = 1):
	low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
	high = constant * np.sqrt(6.0 / (fan_in + fan_out))
	return tf.random_uniform((fan_in, fan_out), low, high, dtype = tf.float32)

class VAE(object):
	"""
		Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
	"""

	def __init__(self, n_in, n_out, sigma_clip=10.):
		self.mu_W = tf.Variable(tf.random_normal([n_in, n_out], mean=0., stddev=0.001), name = 'vae/mu_W')
		self.mu_bias = tf.Variable(tf.zeros([n_out]), name = 'vae/mu_bias')
		self.sigma_W = tf.Variable(tf.random_normal([n_in, n_out], mean=0., stddev=0.001), name = 'vae/sigma_W')
		self.sigma_bias = tf.Variable(tf.zeros([n_out]), name = 'vae/sigma_bias')
		self.sigma_clip= sigma_clip

	def __call__(self, x):
		z_mean = tf.nn.xw_plus_b(x, self.mu_W, self.mu_bias) # b x n_out
		z_log_sigma_sq = tf.nn.xw_plus_b(x, self.sigma_W, self.sigma_bias) # b x n_out
		## apply relu as an clip to avoid -inf and nan issue
		z_log_sigma_sq_l = tf.nn.relu(z_log_sigma_sq + self.sigma_clip) - self.sigma_clip
#		z_log_sigma_sq_h = tf.nn.relu(self.sigma_clip - z_log_sigma_sq_l) + self.sigma_clip
		z_sigma_sq = tf.exp(z_log_sigma_sq_l) # b x n_out
		eps = tf.random_normal(tf.shape(z_mean), 0., 1.) # b x n_out
		z = tf.add(z_mean, tf.multiply(tf.sqrt(z_sigma_sq), eps)) # b x n_out
		latent_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_sigma_sq - \
			tf.square(z_mean) - z_sigma_sq, axis=1)) # 1 
		return latent_loss, z, z_mean, z_log_sigma_sq, z_sigma_sq, eps
