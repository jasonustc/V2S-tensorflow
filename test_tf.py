#! /usr/bin/env python
import tensorflow as tf

a = tf.Variable(tf.random_uniform([5, 2, 3], -1, 1), name='a')
b = tf.Variable(tf.random_uniform([3, 3], -1, 1), name='b')
c = tf.scan(lambda c, x: tf.matmul(x, b), a)
with tf.device("/cpu:0"):
    sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
print sess.run(c)


