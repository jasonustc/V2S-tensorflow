import tensorflow as tf
import numpy as np
import pandas as pd


class RBM(object):
	"""
		The TF implementation of Restricted Boltzmann Machine
	"""

	def __init__(self, num_visible, num_hidden, gibbs_steps=1, learning_rate=0.01):
		self.num_visible = num_visible
		self.num_hidden = num_hidden
		self.gibbs_steps = gibbs_sampling_steps
		self.learning_rate = learning_rate
		self.W = tf.Variable(tf.random_uniform([num_visible, num_hidden], -0.1, 0.1), name = 'rbm_W')
		self.bv = tf.Variable(tf.zeros([num_visible]), name = 'rbm_bv')
		self.bh = tf.Variable(tf.zeros([num_hidden]), name = 'rbm_bh')

	def sample(self, probs):
		#takes in a vector of probabilities, and return a random vector of 0s and 1s
		return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

	def gibbs_sample(self, x):
		# Runs a k-step gibbs chain to sample from the probability distribution of the 
		# RBM defined by W, bh, bv
		def gibbs_step(count, xk):
			# Returns a single gibbs step. The visible values are initialized to xk
			hk = self.sample(tf.sigmoid(tf.nn.xw_plus_b(xk, self.W, self.bh)))#propagate up
			xk = self.sample(tf.sigmoid(tf.nn.xw_plus_b(hk, tf.transpose(self.W), self.bh)))
			return count + 1, xk
		# Run gibbs steps for k iterations
		ct = tf.constant(0) #counter
		# back propagation is not allowed for this loop
		[_, _, x_sample] = tf.while_loop(ct < self.gibbs_steps, 
			[ct, tf.constant(self.gibbs_steps), x], back_prop = False)
		# stop tensorflow from propagating gradients back through the gibbs step
		x_sample = tf.stop_gradients(x_sample)
		return x_sample

	def get_free_energy_cost(self, x):
		# we use this loss in training to get the cost of RBM.
		# draw a sample from the RBM
		x_sample = self.gibbs_sample(x)

		def F(xx):
			# The function computes the free energy of the visible input
			return -tf.reduce_sum(tf.log(1 + tf.exp(tf.nn.xw_plus_b(xx, self.W, self.bh))), axis=1))) - 
				tf.matmul(xx, self.bv, transpose_b = True)
		## the cost is based on the difference in free energy between x and x_sample
		cost = tf.reduce_mean(tf.sub(F(x), F(x_sample)))
		return cost

	def get_cd_update(self, x):
		# contrastive divergence algorithm

		# First, we get the samples of x and h from the probability distribution
		# The sample of x
		x_sample = self.gibbs_sample(x)

		# The sample of hidden nodes, starting from the visible state of x
		h = self.sample(tf.sigmoid(tf.xw_plus_b(x, self.W, self.bh)))
		# The sample of the hidden nodes, starting from the visible state of x_sample
		h_sample = self.sample(tf.sigmoid(tf.xw_plus_b(x_sample, self.W, self.bh)))

		# Next, we update the value of W, bh and bv, based on the differences between the samples
		# that we drew and original values
		lr = tf.constant(self.learning_rate, tf.float32) # the CD learning rate
		size_bt = tf.cast(tf.shape(x)[0], tf.float32) # batch size
		W_ = tf.multiply(lr / size_bt, tf.sub(tf.matmul(x, h, transpose_a = True)), 
			tf.matmul(x_sample, h_sample, transpose_a = True))
		bv_ = tf.multiply(lr / size_bt, tf.reduce_sum(tf.sub(x, x_sample), axis=0, keep_dims=True))
		bh_ = tf.multiply(lr / size_bt, tf.reduce_sum(tf.sub(h, h_sample), axis=0, keep_dims=True))

		# When we do sess.run(update), tf will run all 3 updates
		update = [self.W.assign_add(W_), self.bv.assign_add(bv_), self.bh.assign_add(bh_)]

		return update

	def __call__(self, x):
		return tf.sigmoid(tf.nn.xw_plus_b(x, self.W, bv))



