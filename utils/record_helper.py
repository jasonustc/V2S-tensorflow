#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import h5py
import os
import pdb
import h5py

### MSVD ###
#resize_height = 36
#resize_width = 64
### MSRVTT ###
resize_height = 64
resize_width = 48

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def write_data_as_record(record_writer, data, encode_data, fname,
	title, video_label, caption_label, caption_id, caption_id_1,
	caption_id_2, caption_id_3, caption_id_4, caption_id_5, frame_data):
	data_np = np.asarray(data.astype(np.float32))
	data = data_np.tostring()
	encode_data_np = np.asarray(encode_data).astype(np.float32)
	encode_data = encode_data_np.tostring()
	video_label_np = np.asarray(video_label).astype(np.int32)
	video_label = video_label_np.tostring()
	caption_label_np = np.asarray(caption_label).astype(np.int32)
	caption_label = caption_label_np.tostring()
	caption_id_np = np.asarray(caption_id).astype(np.int32)
	caption_id = caption_id_np.tostring()
	caption_id_1_np = np.asarray(caption_id_1).astype(np.int32)
	caption_id_1 = caption_id_1_np.tostring()
	caption_id_2_np = np.asarray(caption_id_2).astype(np.int32)
	caption_id_2 = caption_id_2_np.tostring()
	caption_id_3_np = np.asarray(caption_id_3).astype(np.int32)
	caption_id_3 = caption_id_3_np.tostring()
	caption_id_4_np = np.asarray(caption_id_4).astype(np.int32)
	caption_id_4 = caption_id_4_np.tostring()
	caption_id_5_np = np.asarray(caption_id_5).astype(np.int32)
	caption_id_5 = caption_id_5_np.tostring()
	frame_data_np = np.asarray(frame_data).astype(np.float32)
	frame_data = frame_data_np.tostring()
	# Example contains a Features proto object
	example = tf.train.Example(
		# Features contains a map of string to Feature proto objects
		features=tf.train.Features(feature={
		# A Feature contains one of either a int64_list,
                # float_list, or bytes_list
		'data': _bytes_feature(data),
		'encode_data': _bytes_feature(encode_data),
		'fname': _bytes_feature(fname),
		'title': _bytes_feature(title),
		'video_label': _bytes_feature(video_label),
		'caption_label': _bytes_feature(caption_label),
		'caption_id': _bytes_feature(caption_id),
		'caption_id_1': _bytes_feature(caption_id_1),
		'caption_id_2': _bytes_feature(caption_id_2),
		'caption_id_3': _bytes_feature(caption_id_3),
		'caption_id_4': _bytes_feature(caption_id_4),
		'caption_id_5': _bytes_feature(caption_id_5),
		'frame_data': _bytes_feature(frame_data)
		}))
	record_writer.write(example.SerializeToString())
#	record_writer.close()
#	print 'data:', data_np[0, 150: 170]
#	print 'encode_data:', encode_data_np[43, 150: 170]
#	print 'video_label:', video_label_np[ :]
#	print 'caption_label:', caption_label_np[ :]
#	print 'caption_id:', caption_id_np[ :]
#	print 'caption_id_1:', caption_id_1_np[ :]
#	print 'caption_id_2:', caption_id_2_np[ :]
#	print 'caption_id_3:', caption_id_3_np[ :]
#	print 'caption_id_4:', caption_id_4_np[ :]
#	print 'caption_id_5:', caption_id_np[ :]

def write_data_as_record_cat_att(record_writer, data, encode_data, fname,
	title, video_label, caption_label, caption_id, caption_id_1,
	frame_data, cat_data, att_data):
	data_np = np.asarray(data.astype(np.float32))
	data = data_np.tostring()
	encode_data_np = np.asarray(encode_data).astype(np.float32)
	encode_data = encode_data_np.tostring()
	video_label_np = np.asarray(video_label).astype(np.int32)
	video_label = video_label_np.tostring()
	caption_label_np = np.asarray(caption_label).astype(np.int32)
	caption_label = caption_label_np.tostring()
	caption_id_np = np.asarray(caption_id).astype(np.int32)
	caption_id = caption_id_np.tostring()
	caption_id_1_np = np.asarray(caption_id_1).astype(np.int32)
	caption_id_1 = caption_id_1_np.tostring()
	frame_data_np = np.asarray(frame_data).astype(np.float32)
	frame_data = frame_data_np.tostring()
	cat_data_np = np.asarray(cat_data).astype(np.float32)
	cat_data = cat_data_np.tostring()
	att_data_np = np.asarray(att_data).astype(np.float32)
	att_data = att_data.tostring()
	# Example contains a Features proto object
	example = tf.train.Example(
		# Features contains a map of string to Feature proto objects
		features=tf.train.Features(feature={
		# A Feature contains one of either a int64_list,
                # float_list, or bytes_list
		'data': _bytes_feature(data),
		'encode_data': _bytes_feature(encode_data),
		'fname': _bytes_feature(fname),
		'title': _bytes_feature(title),
		'video_label': _bytes_feature(video_label),
		'caption_label': _bytes_feature(caption_label),
		'caption_id': _bytes_feature(caption_id),
		'caption_id_1': _bytes_feature(caption_id_1),
		'frame_data': _bytes_feature(frame_data),
		'cat_data': _bytes_feature(cat_data),
		'att_data': _bytes_feature(att_data)
		}))
	record_writer.write(example.SerializeToString())

def read_record_file(filename):
	assert os.path.isfile(filename)
	record_iterator = tf.python_io.tf_record_iterator(path=filename)
	for string_record in record_iterator:
		example = tf.train.Example()
		example.ParseFromString(string_record)
		data_string = example.features.feature['data'].bytes_list.value[0]
		feat = np.fromstring(data_string, dtype=np.float32)
		print feat[100:200]

def read_and_decode(filename):
	# why num_epoches not work here?
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		features = {
		'data': tf.FixedLenFeature([], tf.string),
		'encode_data': tf.FixedLenFeature([], tf.string),
		'fname': tf.VarLenFeature(tf.string),
		'title': tf.VarLenFeature(tf.string),
		'video_label': tf.FixedLenFeature([], tf.string),
		'caption_label': tf.FixedLenFeature([], tf.string),
		'caption_id': tf.FixedLenFeature([], tf.string),
		'caption_id_1': tf.FixedLenFeature([], tf.string),
		'caption_id_2': tf.FixedLenFeature([], tf.string),
		'caption_id_3': tf.FixedLenFeature([], tf.string),
		'caption_id_4': tf.FixedLenFeature([], tf.string),
		'caption_id_5': tf.FixedLenFeature([], tf.string)
		})
	data = tf.decode_raw(features['data'], tf.float32)
	data = tf.reshape(data, [45, 8192])
	encode_data = tf.decode_raw(features['encode_data'], tf.float32)
	encode_data = tf.reshape(encode_data, [45, 8192])
	video_label = tf.decode_raw(features['video_label'], tf.int32)
	video_label = tf.reshape(video_label, [45])
	caption_label = tf.decode_raw(features['caption_label'], tf.int32)
	caption_label = tf.reshape(caption_label, [35])
	caption_id = tf.decode_raw(features['caption_id'], tf.int32)
	caption_id = tf.reshape(caption_id, [35])
	caption_id_1 = tf.decode_raw(features['caption_id_1'], tf.int32)
	caption_id_1 = tf.reshape(caption_id_1, [35])
	caption_id_2 = tf.decode_raw(features['caption_id_2'], tf.int32)
	caption_id_2 = tf.reshape(caption_id_2, [35])
	caption_id_3 = tf.decode_raw(features['caption_id_3'], tf.int32)
	caption_id_3 = tf.reshape(caption_id_3, [35])
	caption_id_4 = tf.decode_raw(features['caption_id_4'], tf.int32)
	caption_id_4 = tf.reshape(caption_id_4, [35])
	caption_id_5 = tf.decode_raw(features['caption_id_5'], tf.int32)
	caption_id_5 = tf.reshape(caption_id_5, [35])
        fname = features['fname']
        title = features['title']
        return data, encode_data, fname, title, video_label, caption_label, caption_id, \
            caption_id_1, caption_id_2, caption_id_3, caption_id_4, caption_id_5

def read_and_decode_with_frame(filename):
	# why num_epoches not work here?
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		features = {
		'data': tf.FixedLenFeature([], tf.string),
		'encode_data': tf.FixedLenFeature([], tf.string),
		'fname': tf.VarLenFeature(tf.string),
		'title': tf.VarLenFeature(tf.string),
		'video_label': tf.FixedLenFeature([], tf.string),
		'caption_label': tf.FixedLenFeature([], tf.string),
		'caption_id': tf.FixedLenFeature([], tf.string),
		'caption_id_1': tf.FixedLenFeature([], tf.string),
		'caption_id_2': tf.FixedLenFeature([], tf.string),
		'caption_id_3': tf.FixedLenFeature([], tf.string),
		'caption_id_4': tf.FixedLenFeature([], tf.string),
		'caption_id_5': tf.FixedLenFeature([], tf.string),
		'frame_data': tf.FixedLenFeature([], tf.string)
		})
	data = tf.decode_raw(features['data'], tf.float32)
	data = tf.reshape(data, [45, 8192])
	encode_data = tf.decode_raw(features['encode_data'], tf.float32)
	encode_data = tf.reshape(encode_data, [45, 8192])
	video_label = tf.decode_raw(features['video_label'], tf.int32)
	video_label = tf.reshape(video_label, [45])
	caption_label = tf.decode_raw(features['caption_label'], tf.int32)
	caption_label = tf.reshape(caption_label, [35])
	caption_id = tf.decode_raw(features['caption_id'], tf.int32)
	caption_id = tf.reshape(caption_id, [35])
	caption_id_1 = tf.decode_raw(features['caption_id_1'], tf.int32)
	caption_id_1 = tf.reshape(caption_id_1, [35])
	caption_id_2 = tf.decode_raw(features['caption_id_2'], tf.int32)
	caption_id_2 = tf.reshape(caption_id_2, [35])
	caption_id_3 = tf.decode_raw(features['caption_id_3'], tf.int32)
	caption_id_3 = tf.reshape(caption_id_3, [35])
	caption_id_4 = tf.decode_raw(features['caption_id_4'], tf.int32)
	caption_id_4 = tf.reshape(caption_id_4, [35])
	caption_id_5 = tf.decode_raw(features['caption_id_5'], tf.int32)
	caption_id_5 = tf.reshape(caption_id_5, [35])
	frame_data = tf.decode_raw(features['frame_data'], tf.float32)
	frame_data = tf.reshape(frame_data, [45, resize_height*resize_width*3])
	fname = features['fname']
	title = features['title']
	return data, encode_data, fname, title, video_label, caption_label, caption_id, \
	caption_id_1, caption_id_2, caption_id_3, caption_id_4, caption_id_5, frame_data
#	with tf.Session() as sess:
#		coord = tf.train.Coordinator()
#		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#		sess.run(tf.global_variables_initializer())
#		sess.run(tf.local_variables_initializer())
#		np_data = sess.run(data)
#		np_label = sess.run(label)
#		np_caption_id = sess.run(caption_id)
#		np_video_label = sess.run(video_label)
#		print 'data:', np_data[0, 100: 200]
#		print 'caption_label:', np_label[ :]
#		print 'caption_id:', np_caption_id[ :]
#		print 'video_label:', np_video_label[ :]
#		pdb.set_trace()
#		coord.request_stop()
#		coord.join(threads)
#		sess.close()

def read_and_decode_frame_cat_att(filename):
	# why num_epoches not work here?
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		features = {
		'data': tf.FixedLenFeature([], tf.string),
		'encode_data': tf.FixedLenFeature([], tf.string),
		'fname': tf.VarLenFeature(tf.string),
		'title': tf.VarLenFeature(tf.string),
		'video_label': tf.FixedLenFeature([], tf.string),
		'caption_label': tf.FixedLenFeature([], tf.string),
		'caption_id': tf.FixedLenFeature([], tf.string),
		'caption_id_1': tf.FixedLenFeature([], tf.string),
		'frame_data': tf.FixedLenFeature([], tf.string),
		'cat_data': tf.FixedLenFeature([], tf.string),
		'att_data': tf.FixedLenFeature([], tf.string)
		})
	data = tf.decode_raw(features['data'], tf.float32)
	data = tf.reshape(data, [45, 8192])
	encode_data = tf.decode_raw(features['encode_data'], tf.float32)
	encode_data = tf.reshape(encode_data, [45, 8192])
	video_label = tf.decode_raw(features['video_label'], tf.int32)
	video_label = tf.reshape(video_label, [45])
	caption_label = tf.decode_raw(features['caption_label'], tf.int32)
	caption_label = tf.reshape(caption_label, [35])
	caption_id = tf.decode_raw(features['caption_id'], tf.int32)
	caption_id = tf.reshape(caption_id, [35])
	caption_id_1 = tf.decode_raw(features['caption_id_1'], tf.int32)
	caption_id_1 = tf.reshape(caption_id_1, [35])
	frame_data = tf.decode_raw(features['frame_data'], tf.float32)
	frame_data = tf.reshape(frame_data, [45, resize_height*resize_width*3])
	cat_data = tf.decode_raw(features['cat_data'], tf.float32)
	cat_data = tf.reshape(cat_data, [20])
	att_data = tf.decode_raw(features['att_data'], tf.float32)
	att_data = tf.reshape(att_data, [1000])
	fname = features['fname']
	title = features['title']
	return data, encode_data, fname, title, video_label, caption_label, caption_id, \
	caption_id_1, frame_data, cat_data, att_data

##### test data reading #####
if __name__ == "__main__":
    data, encode_data, fname, title, video_label, caption_label, caption_id, \
    caption_id_1, frame_data, cat_data, att_data = read_and_decode_frame_cat_att('/home/shenxu/data/msrvtt_frame_cat_att/train.tfrecords')
    with tf.device("/cpu:0"):
        sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    data, encode_data, fname, title, video_label, caption_label, caption_id, caption_id_1, frame_data, cat_data, att_data = \
        sess.run([data, encode_data, fname, title, video_label, caption_label, caption_id, caption_id_1, frame_data, cat_data, att_data])
    coord.request_stop()
    coord.join(threads)
    pdb.set_trace()

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
