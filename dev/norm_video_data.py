#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import pdb
import time
import json
from collections import defaultdict
#from keras.preprocessing import sequence
from cocoeval import COCOScorer
import unicodedata
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from modules.variational_autoencoder import VAE
from utils.model_ops import *
from utils.record_helper import read_and_decode
import random

batch_size = 50
video_data_path_train = '/disk_2T/shenxu/msrvtt_feat_vgg_c3d_batch/train.tfrecords'
n_train_samples = 130175

def norm():
    assert os.path.isfile(video_data_path_train)
    # preprocess on the CPU
    with tf.device('/cpu:0'):
        train_data, _, _, _, _, _, _, _, _, _, _, _ = read_and_decode(video_data_path_train)
       # batches
        train_data = tf.train.batch([train_data], batch_size=batch_size, num_threads=1, capacity=batch_size)

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    # initialize epoch variable in queue reader
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tStart_total = time.time()
    n_epoch_steps = int(n_train_samples / batch_size)
    global_max_feat = np.zeros((8192,)).astype(np.float32)
    global_min_feat = np.zeros((8192,)).astype(np.float32) + 1000.
    for step in range(n_epoch_steps):
        if step % 10 == 0:
            print step, ' / ', n_epoch_steps
        train_feat = sess.run(train_data) # 100 * 45 * 8192
        max_feat = np.amax(train_feat, axis=(0,1))
        min_feat = np.amin(train_feat, axis=(0,1))
        global_max_feat = np.maximum(global_max_feat, max_feat)
        global_min_feat = np.minimum(global_min_feat, min_feat)

    np.savez('msrvtt_max_feat', global_max_feat=global_max_feat, global_min_feat=global_min_feat)
    print 'global_max_values:'
    print global_max_feat[0:100]
    print global_max_feat[-100:]
    print 'global_min_values:'
    print global_min_feat[0:100]
    print global_min_feat[-100:]
    coord.request_stop()
    coord.join(threads)
    tStop_total = time.time()
    print "Total Time Cost:", round(tStop_total - tStart_total,2), "s"
    sess.close()

if __name__ == '__main__':
    norm()
