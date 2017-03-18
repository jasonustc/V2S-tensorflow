#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import h5py
import os
import pdb

def fake_hdf5_data(data_folder):
	assert os.path.isdir(data_folder)
	"""
	data = np.zeros((100, 45, 8192), dtype=np.float)
	title = np.array(['this is a video name' for i in xrange(100)])
	caption_id = np.array([i * 3 for i in xrange(100)])
	video_label = np.ones((100, 45), dtype=np.float)
	caption_label = np.ones((100, 35), dtype=np.float)
	paths = []
	for i in xrange(3):
		name = 'video' + str(i) + '.h5'
		batch = h5py.File(os.path.join(data_folder, name), 'w')
		batch['data'] = data
		batch['title'] = title
		batch['caption_id'] = caption_id
		batch['caption_label'] = caption_label
		batch['video_label'] = video_label
		paths.append(os.path.join(data_folder, name))
	"""

	index = h5py.File('index.h5', 'w')
	index['names'] = np.array([os.path.join(data_folder, 'train000001.h5')])

fake_hdf5_data('data0')



#a = tf.Variable(tf.random_uniform([5, 2, 3], -1, 1), name='a')
#b = tf.Variable(tf.random_uniform([3, 3], -1, 1), name='b')
#c = tf.scan(lambda c, x: tf.matmul(x, b), a)
#with tf.device("/cpu:0"):
#    sess = tf.InteractiveSession()
#tf.initialize_all_variables().run()
#print sess.run(c)


#tStart = time.time()
#current_batch = h5py.File('/home/shenxu/data/msvd_feat_vgg_c3d_batch/test000096.h5')
#current_feats = np.array(current_batch['data'])
#current_video_masks = np.array(current_batch['video_label'])
#current_caption_matrix = np.array(current_batch['caption_id'])
#current_caption_masks = np.array(current_batch['caption_label'])
#tEnd = time.time()
#print 'data reading time:', round(tEnd - tStart, 2), 's'
