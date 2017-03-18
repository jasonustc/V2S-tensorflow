#! /usr/bin/env python
#################################################################################
#     File Name           :     make_path.py
#     Created By          :     shenxu
#     Creation Date       :     [2016-09-06 22:42]
#     Last Modified       :     [2017-02-25 22:36]
#     Description         :      
#################################################################################
import os

def change_path(folder, out_file_path):
    assert os.path.isdir(folder)
    current_path = os.getcwd()
    out = open(out_file_path, 'w')
    file_list = os.listdir(folder)
    for line in file_list:
        if line[-2:] == 'h5':
            line = current_path + '/' + folder + '/' + line
            out.write(line + '\n')
    out.close()

change_path('data0/train_batches','data0/train_vn.txt')
change_path('data0/val_batches','data0/val_vn.txt')


