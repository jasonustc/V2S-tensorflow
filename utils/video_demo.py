from feat_extractor import FeatExtractor
from caffe_io import load_video_from_list
from caffe_io import save_as_hdf5

caffe_root = '/home/shenxu/caffe-multigpu/'

import os
import sys
import argparse
import time
import json
import pdb
import h5py

sys.path.insert(0, caffe_root + 'python')
import caffe

import numpy as np

def get_frame_list(list_file):
  assert os.path.isfile(list_file)
  dat = json.load(open(list_file))
  return dat[0]

def get_list(folder, key):
  assert os.path.isdir(folder)
  pylist = []
  for roots, dirs, files in os.walk(folder):
    for ele in files:
      if ele.endswith(key):
        pylist.append(roots + '/' + ele)
  return pylist

frame_id_folder = '/disk_new/shenxu/msvd_data/'

def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'video_folder',
    help = 'Input video folder.')
  parser.add_argument(
    'output_dir',
    help = 'Output directory.')
  parser.add_argument(
    '--sample_rate',
    type = float,
    default = 5.0,
    help = 'Number of frames sampled per second')
  parser.add_argument(
    '--model_def',
    default = '/home/shenxu/V2S-tensorflow/models/VGG_ILSVRC_19_layers_deploy.prototxt',
    help = 'Model definition file (default VGG19)')
  parser.add_argument(
    '--pretrained_model',
    default = '/home/shenxu/V2S-tensorflow/models/VGG_ILSVRC_19_layers.caffemodel',
    help = 'Model parameter file (default VGG19)')
  parser.add_argument(
    '--layers',
    default = 'fc6,fc7',
    help = 'Layers to be extracted, separated by commas')
  parser.add_argument(
    '--cpu',
    action = 'store_true',
    help = 'Use CPU if set')
  parser.add_argument(
    '--oversample',
    action = 'store_true',
    help = 'Oversample 10 patches per frame if set')
  args = parser.parse_args()
  if args.cpu:
    caffe.set_mode_cpu()
    print 'CPU mode'
  else:
    caffe.set_mode_gpu()
    print 'GPU mode'
  oversample = False
  if args.oversample:
    oversample = True
  extractor = FeatExtractor(args.model_def, args.pretrained_model, oversample=oversample)
  blobs = args.layers.split(',')
#  with open(args.video_list) as f:
#    videos = [l.rstrip() for l in f]
  list_files = get_list(frame_id_folder, 'json')
  for list_file in list_files:
    frame_list = get_frame_list(list_file)
    video_path = args.video_folder + '/' + os.path.basename(list_file).split('.')[0] + '.avi'
    frames = load_video_from_list(video_path, frame_list)
    if len(frames) < 1: # failed to open the video
      continue
    start = time.time()
    feats = extractor.extract_batch(frames, blobs)
    print '%s feature extracted in %f seconds.' % (os.path.basename(video_path), time.time()-start)
    # save the features
    save_as_hdf5(feats, os.path.join(args.output_dir,
        '%s.h5' % os.path.basename(video_path).split('.')[0]))
  return

if __name__ == '__main__':
  main(sys.argv)
