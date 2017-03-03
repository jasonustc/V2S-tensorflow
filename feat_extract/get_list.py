#! /usr/bin/env python
#################################################################################
#     File Name           :     get_list.py
#     Created By          :     shenxu
#     Creation Date       :     [2017-03-01 09:07]
#     Last Modified       :     [2017-03-02 06:39]
#     Description         :      
#################################################################################
import os
import pdb
import sys
sys.path.append('../hdf5_generator')
from generate_nolabel import splitdata
import json

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
        video_path = video_folder + os.path.basename(path).split('.')[0] + '.avi'
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

if __name__ == '__main__':
#    splitdata('/disk_new/XuKS/YouTubeClips', 1200, 100)
    get_index_file_c3d('/disk_new/shenxu/msvd_data/', '/disk_new/XuKS/YouTubeClips/',
            '/disk_new/shenxu/msvd_feat_c3d/', 'c3d_input_list.txt', 'c3d_output_prefix.txt')
