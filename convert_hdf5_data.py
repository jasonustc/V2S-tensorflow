import h5py
import numpy as np
import glob
import os
import pdb

def get_hdf5_data(h5_file):
	train_batch = h5py.File(h5_file)
	cont = train_batch['cont']
	fc7 = train_batch['frame_fc7']
	pdb.set_trace()
