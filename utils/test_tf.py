#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import h5py
import os
import pdb

def test_multi_gpu():
    with tf.device('/gpu:0'):
        a = tf.placeholder(tf.float32, [10, 10])
        b = tf.placeholder(tf.float32, [10, 3])
        c = tf.matmul(a, b)
        l1 = tf.reduce_sum(a)

    with tf.device('/gpu:1'):
        d = tf.reduce_sum(c, axis=1)
        e = tf.reduce_sum(d)
        l2 = e

    with tf.device('/cpu:0'):
        l = l1 + l2
    return l, l1, l2, a, b

A = np.random.rand(10, 10).astype('float32')
B = np.random.rand(10, 3).astype('float32')
tf_l, tf_l1, tf_l2, tf_a, tf_b = test_multi_gpu()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    l,l1,l2 = sess.run([tf_l, tf_l1, tf_l2], feed_dict={tf_a: A, tf_b: B})
    print l, l1, l2
