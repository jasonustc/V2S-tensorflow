#! /usr/bin/env python
import tensorflow as tf
import time
import h5py
import numpy as np

#a = tf.Variable(tf.random_uniform([5, 2, 3], -1, 1), name='a')
#b = tf.Variable(tf.random_uniform([3, 3], -1, 1), name='b')
#c = tf.scan(lambda c, x: tf.matmul(x, b), a)
#with tf.device("/cpu:0"):
#    sess = tf.InteractiveSession()
#tf.initialize_all_variables().run()
#print sess.run(c)


tStart = time.time()
current_batch = h5py.File('/home/shenxu/data/msvd_feat_vgg_c3d_batch/test000096.h5')
current_feats = np.array(current_batch['data'])
current_video_masks = np.array(current_batch['video_label'])
current_caption_matrix = np.array(current_batch['caption_id'])
current_caption_masks = np.array(current_batch['caption_label'])
tEnd = time.time()
print 'data reading time:', round(tEnd - tStart, 2), 's'
