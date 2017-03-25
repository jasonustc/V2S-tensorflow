#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import h5py
import os
import pdb
import h5py

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def write_data_as_record(h5_file):
	assert os.path.isfile(h5_file)
	batch = h5py.File(h5_file)
	feat_np = np.asarray(batch['data']).astype(np.float32)[:2, :2, :]
	feat = feat_np.tostring()
	label_np = np.asarray(batch['caption_label']).astype(np.int32)[:2, :]
	label = label_np.tostring()
	caption_id_np = np.asarray(batch['caption_id']).astype(np.int32)[:2, :]
	caption_id = caption_id_np.tostring()
	video_label_np = np.asarray(batch['video_label']).astype(np.int32)[:2, :]
	video_label = video_label_np.tostring()
	filename = os.path.splitext(h5_file)[0] + '.tfrecords'
	print('Writing', filename)
	writer = tf.python_io.TFRecordWriter(filename)
	# Example contains a Features proto object
	example = tf.train.Example(
		# Features contains a map of string to Feature proto objects
		features=tf.train.Features(feature={
		# A Feature contains one of either a int64_list,
        # float_list, or bytes_list
		'data': _bytes_feature(feat),
		'label': _bytes_feature(label),
		'caption_id': _bytes_feature(caption_id),
		'video_label': _bytes_feature(video_label)
		}))
	writer.write(example.SerializeToString())
	writer.close()
	print 'data:',feat_np[0,0, 100: 200]
	print 'label:', label_np[0, :]
	print 'caption_id:', caption_id_np[0, :]
	print 'video_label:', video_label_np[0, :]
	return filename

def read_record_file(filename):
	assert os.path.isfile(filename)
	record_iterator = tf.python_io.tf_record_iterator(path=filename)
	for string_record in record_iterator:
		example = tf.train.Example()
		example.ParseFromString(string_record)
		data_string = example.features.feature['data'].bytes_list.value[0]
		feat = np.fromstring(data_string, dtype=np.float32)
		print feat[100:200]

def read_record_queue(filename):
	# why num_epoches not work here?
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		features = {
		'data': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.string),
		'caption_id': tf.FixedLenFeature([], tf.string),
		'video_label': tf.FixedLenFeature([], tf.string)
		})
	data = tf.decode_raw(features['data'], tf.float32)
	data = tf.reshape(data, [2, 2, 8192])
	label = tf.decode_raw(features['label'], tf.int32)
	label = tf.reshape(label, [2, 35])
	caption_id = tf.decode_raw(features['caption_id'], tf.int32)
	caption_id = tf.reshape(caption_id, [2, 35])
	video_label = tf.decode_raw(features['video_label'], tf.int32)
	video_label = tf.reshape(video_label, [2, 45])
	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		np_data = sess.run(data)
		np_label = sess.run(label)
		np_caption_id = sess.run(caption_id)
		np_video_label = sess.run(video_label)
		print 'data:', np_data[0,0, 100: 200]
		print 'label:', np_label[0, :]
		print 'caption_id:', np_caption_id[0, :]
		print 'video_label:', np_video_label[0, :]
		pdb.set_trace()
		coord.request_stop()
		coord.join(threads)
		sess.close()

record_file=write_data_as_record('../data0/train000000.h5')
#read_record_file(record_file)
read_record_queue(record_file)




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
