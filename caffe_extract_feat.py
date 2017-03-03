#! /usr/bin/env python
#################################################################################
#     File Name           :     caffe_extract_feat.py
#     Created By          :     shenxu
#     Creation Date       :     [2017-02-28 22:38]
#     Last Modified       :     [2017-02-28 23:00]
#     Description         :     use caffe for feature extraction of frames
#################################################################################
import pdb
import sys
import cv2

caffe_root = '/home/shenxu/caffe-multigpu/'

sys.path.insert(0, caffe_root + 'python')

import caffe

def get_frame_data(clip_path, frame_ids):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print "could not open: ", clip_path
        sys.exit()
    for frame_id in frame_ids:
