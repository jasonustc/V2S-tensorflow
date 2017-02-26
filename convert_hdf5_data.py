import h5py
import numpy as np
import glob
import os
import pdb

def get_hdf5_data(h5_file, fname):
    assert os.path.isfile(h5_file)
    train_batch = h5py.File(h5_file)
    data = train_batch[fname]
    print fname, 'shape:', data.shape
    return data

def split_data(fc7, title):
    fc7 = np.array(fc7)
    title = np.array(title)
    batch_data = []
    num = fc7.shape[0] / 48
    cont_data = np.zeros((48, fc7.shape[1])) - 1
    cont_data[-1, :] = 0
    for i in xrange(num):
        batch_data.append((fc7[i * 48 : (i+1) * 48], cont_data, title[i*48 : (i+1)*48]))
    return batch_data

def write_hdf5_data(h5_data, outp_path):
    with h5py.File(outp_path, 'w') as hf:
        hf.create_dataset('label', data = h5_data['label'])
        hf.create_dataset('data', data = h5_data['data'])
        hf.create_dataset('title', data = h5_data['title'])

if __name__ == '__main__':
    fc7 = get_hdf5_data('/home/shenxu/data/hdf5_video_vgg_fc6/train_batches/batch_90.h5', 'frame_fc7')
    title = get_hdf5_data('/home/shenxu/data/hdf5_sentence_vocab_vtt_buff16/train_batches/batch_90.h5', 'target_sentence')
    batch_data = split_data(fc7, title)
    for idx, ele in enumerate(batch_data):
        h5_data = {}
        h5_data['data'] = ele[0]
        h5_data['label'] = ele[1]
        h5_data['title'] = ele[2]
        write_hdf5_data(h5_data, 'test_data/' + str(idx) + '.h5')
    fc7 = get_hdf5_data('test_data/3.h5', 'data')
    title = get_hdf5_data('test_data/7.h5', 'title')
    label = get_hdf5_data('test_data/19.h5', 'label')
    pdb.set_trace()
