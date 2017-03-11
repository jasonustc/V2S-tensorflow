#! /usr/bin/env python
#################################################################################
#     File Name           :     get_list.py
#     Created By          :     shenxu
#     Creation Date       :     [2017-03-01 09:07]
#     Last Modified       :     [2017-03-07 15:37]
#     Description         :      
#################################################################################
import os
import pdb
import sys
sys.path.append('../hdf5_generator')
from generate_nolabel import splitdata
import json
import numpy as np
import struct
import h5py

def get_list(path, key):
    assert os.path.isdir(path)
    pylist = []
    for roots, dirs, files in os.walk(path):
        for ff in files:
            if ff.endswith(key):
                pylist.append(roots + '/' + ff)
    return pylist


def write_list(pylist, path):
    with open(path, 'w') as of:
        for ff in pylist:
            of.write(ff + '\n')
        of.close()

def get_index_file_c3d(video_data_folder, video_folder, c3d_feat_folder, input_list,
        output_prefix):
    assert os.path.isdir(video_data_folder)
    assert os.path.isdir(video_folder)
    assert os.path.isdir(c3d_feat_folder)
    of_input = open(input_list, 'w')
    of_output = open(output_prefix, 'w')
    for path in os.listdir(video_data_folder):
        video_data_path = video_data_folder + path
        video_path = video_folder + os.path.basename(path).split('.')[0] + '.mp4'
        frame_data = json.load(open(video_data_path))
        frame_list = frame_data[0]
        num = 0
        for start, end in frame_list:
            if start == 0:
                start_frame = 0
            elif end - start < 8:
                start_frame = int(end - 16)
            else:
                start_frame = int(start - 8)
            of_input.write(video_path + ' ' + str(start_frame) + ' ' + str(0) + ' \n')
            output_prefix = c3d_feat_folder + os.path.basename(path).split('.')[0] + '/' + '{:06d}'.format(num)
            if not os.path.isdir(os.path.dirname(output_prefix)):
                os.mkdir(os.path.dirname(output_prefix))
            of_output.write(output_prefix + '\n')
            num += 1
    of_input.close()
    of_output.close()

def get_binary_data_c3d(input_file, shape, dtype = np.float32):
    with open(input_file, 'rb') as f:
        num = struct.unpack("i", f.read(4))
        ch = struct.unpack("i", f.read(4))
        length = struct.unpack("i", f.read(4))
        height = struct.unpack("i", f.read(4))
        width = struct.unpack("i", f.read(4))
        data = np.fromfile(f, dtype = dtype)
        array = np.reshape(data, shape)
        return array

def merge_and_save_as_h5(feat_folder, feat_name, key):
    assert os.path.isdir(feat_folder)
    feats = os.listdir(feat_folder)
    batch_data = []
    for feat_file in feats:
        if feat_file.endswith(feat_name):
            frame_data = get_binary_data_c3d(os.path.join(feat_folder, feat_file), [1, 4096])
            batch_data.append(frame_data)
    all_data = np.vstack(batch_data)
    h5_file = feat_folder[:feat_folder.rfind('/')] + feat_folder[feat_folder.rfind('/'):] +  \
        '_c3d_' + feat_name + '.h5'
    batch = h5py.File(h5_file, 'w')
    batch[key] = all_data

def build_list(batch_folder):
    assert os.path.isdir(batch_folder)
    datasets = ['train', 'val', 'test']
    for dataset in datasets:
        with open(os.path.join(batch_folder, dataset + '_vn.txt'), 'w') as of:
            for f in os.listdir(batch_folder):
                if f.startswith(dataset) and f.endswith('.h5'): 
                    of.write(os.path.join(batch_folder, f) + '\n')
            of.close()


if __name__ == '__main__':
#    splitdata('/disk_new/XuKS/YouTubeClips', 1200, 100)
#    get_index_file_c3d('/disk_2T/shenxu/msvd_data/', '/disk_2T/XuKS/YouTubeClips_convert/',
#            '/disk_2T/shenxu/msvd_feat_c3d/', 'c3d_input_list.txt', 'c3d_output_prefix.txt')
#    get_binary_data_c3d('/disk_2T/shenxu/msvd_feat_c3d/n_Z0-giaspE_270_278/000000.fc6-1', [1, 4096])
#    c3d_feat_folder = '/disk_2T/shenxu/msvd_feat_c3d/'
#    folders = os.listdir(c3d_feat_folder)
#    count = 0
#    for folder in folders:
#        if os.path.isdir(os.path.join(c3d_feat_folder, folder)):
#            merge_and_save_as_h5(os.path.join(c3d_feat_folder, folder), 'fc6-1', 'data')
#            merge_and_save_as_h5(os.path.join(c3d_feat_folder, folder), 'fc7-1', 'data')
#            count += 1
#            if count % 100 == 0:
#                print count
#    print count
    build_list('/home/shenxu//data/msvd_feat_vgg_c3d_batch')
