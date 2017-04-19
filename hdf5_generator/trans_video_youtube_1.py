import h5py
import numpy as np
import json
import pdb
import unicodedata
import glob
import os
from keras.preprocessing import sequence
import random
import sys
sys.path.insert(0, os.path.abspath('../'))
from utils.record_helper import write_data_as_record
import tensorflow as tf
import cv2

#feature_folder = '/disk_2T/shenxu/msrvtt_feat_vgg_c3d_batch/'
#vgg_feat_folder_train = '/disk_new/XuKS/caffe-recurrent/examples/s2vt/hdf5_vgg_train/'
#vgg_feat_folder_val = '/disk_new/XuKS/caffe-recurrent/examples/s2vt/hdf5_vgg_val/'
#vgg_feat_folder_test = '/disk_new/XuKS/caffe-recurrent/examples/s2vt/hdf5_vgg_test/'
#c3d_feat_folder_train = '/disk_new/XuKS/caffe-recurrent/examples/s2vt/hdf5_c3d_train/'
#c3d_feat_folder_val = '/disk_new/XuKS/caffe-recurrent/examples/s2vt/hdf5_c3d_val/'
#c3d_feat_folder_test = '/disk_new/XuKS/caffe-recurrent/examples/s2vt/hdf5_c3d_test/'
#v2s_json = 'msrvtt2sent.json'
#vgg_feat_name = 'frame_vgg'
#c3d_feat_name = 'frame_c3d'
home_folder = '/home/shenxu/V2S-tensorflow/'
vgg_feat_name = 'fc6'
c3d_feat_name = 'fc6'
feature_folder = '/disk_2T/shenxu/msvd_feat_vgg_c3d_frame/'
vgg_feat_folder = '/disk_2T/shenxu/msvd_feat_vgg/'
c3d_feat_folder = '/disk_2T/shenxu/msvd_feat_c3d/'
word_count_threshold = 1
v2s_json = home_folder + 'msvd2sent.json'
video_frame_folder = '/disk_2T/shenxu/YoutubeClip_frames/'
resize_height = 36
resize_width = 64

def get_cap_ids(title, wordtoix, cap_length, pad='post'):
    assert pad in ['pre', 'post']
    cap_id = [wordtoix[word] for word in title.lower().split(' ') if word in wordtoix]
    n_words = len(cap_id) if len(cap_id) < cap_length else cap_length - 1
    cap_id = [cap_id]
    ## the last word must be '<END>' ###
    if pad == 'post':
        return sequence.pad_sequences(cap_id, padding='post', maxlen=cap_length-1), n_words
    else:
        return sequence.pad_sequences(cap_id, padding='pre', maxlen=cap_length-1), n_words

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

def build_vocab(train_set, dataset_name):
    re = json.load(open(v2s_json))
    print 'load json from', v2s_json
    train_title = []
    for video in train_set:
        for title in re[video]:
            train_title.append(title)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(train_title, word_count_threshold)
    np.save(dataset_name + '_wordtoix', wordtoix)
    np.save(dataset_name + '_ixtoword', ixtoword)
    np.save(dataset_name + '_bias_init_vector', bias_init_vector)
    return wordtoix, ixtoword

def  load_frame(frame_path, resize_height=None, resize_width=None):
    assert os.path.isfile(frame_path)
    frame_data = cv2.imread(frame_path)
#    cv2.imwrite('test.jpg', frame_data)
#    print frame_data.shape
    if resize_height and resize_width:
        frame_data = cv2.resize(frame_data, (resize_width, resize_height))
#        cv2.imwrite('test_resize.jpg', frame_data)
#        print frame_data.shape
    return np.reshape(frame_data, (resize_width*resize_height*3,))

def load_video_frames(video_path, resize_height=None, resize_width=None):
    assert os.path.isdir(video_path)
    frames = os.listdir(video_path)
    frame_data = []
    for frame in frames:
        frame_data.append(load_frame(os.path.join(video_path, frame), resize_height=resize_height, resize_width=resize_width))
    return np.stack(frame_data)

def trans_video_youtube_record(datasplit_list, datasplit, vgg_feat_name,
        c3d_feat_name, wordtoix):
    assert len(datasplit_list) > 0
    re = json.load(open(home_folder + 'msvd2sent.json'))
    n_length = 45
    cap_length = 35

    initial = 0
    cnt = 0
    filename = feature_folder + datasplit + '.tfrecords'
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for ele in datasplit_list:
        vgg_feat_file = vgg_feat_folder + ele + '.h5'
        c3d_feat_file = c3d_feat_folder + ele + '_c3d_' + c3d_feat_name + '-1.h5'
        frame_folder = video_frame_folder + ele
        assert os.path.isfile(vgg_feat_file)
        assert os.path.isfile(c3d_feat_file)
        assert os.path.isdir(frame_folder)
        vgg_feat = np.squeeze(np.asarray(h5py.File(vgg_feat_file)[vgg_feat_name]))
        c3d_feat = np.squeeze(np.asarray(h5py.File(c3d_feat_file)['data']))
        frame_feat = load_video_frames(frame_folder, resize_height=resize_height, resize_width=resize_width)
        # to solve the number of frames mismatch between different videos
        sample_data = np.zeros([n_length, 4096 * 2])
        encode_data = np.zeros([n_length, 4096 * 2])
        frame_data = np.zeros([n_length, resize_height*resize_width*3])
        concat_feat = np.concatenate((vgg_feat, c3d_feat), axis = 1)
        # post pad
        sample_data[:concat_feat.shape[0], :] = concat_feat
        frame_data[:frame_feat.shape[0], :] = frame_feat
        # pre pad
        encode_data[-concat_feat.shape[0]:, :] = concat_feat
        video_name = ele
        if video_name in re.keys():
            print video_name, 'num_sen:', len(re[video_name])
            caps = re[video_name]
            for xxx in caps:
                en_cap_ind = random.sample(range(0, len(caps)), 5)
                if len(xxx.split(' ')) < 35:
                    title = unicodedata.normalize('NFKD', xxx).encode('ascii','ignore')
                    ### video label ###
                    vl = np.zeros([n_length])
                    vl[:concat_feat.shape[0]] = 1
                    ### caption of word ids ###
                    cap_id, n_words = get_cap_ids(xxx, wordtoix, cap_length, pad='post')
                    cap_id_1, n_word_1 = get_cap_ids(caps[en_cap_ind[0]], wordtoix, cap_length, pad='pre')
                    cap_id_2, n_word_2 = get_cap_ids(caps[en_cap_ind[1]], wordtoix, cap_length, pad='pre')
                    cap_id_3, n_word_3 = get_cap_ids(caps[en_cap_ind[2]], wordtoix, cap_length, pad='pre')
                    cap_id_4, n_word_4 = get_cap_ids(caps[en_cap_ind[3]], wordtoix, cap_length, pad='pre')
                    cap_id_5, n_word_5 = get_cap_ids(caps[en_cap_ind[4]], wordtoix, cap_length, pad='pre')
                    cap_id = np.hstack([cap_id, np.zeros([1,1])]).astype(int)
                    cap_id_1 = np.hstack([cap_id_1, np.zeros([1,1])]).astype(int)
                    cap_id_2 = np.hstack([cap_id_2, np.zeros([1,1])]).astype(int)
                    cap_id_3 = np.hstack([cap_id_3, np.zeros([1,1])]).astype(int)
                    cap_id_4 = np.hstack([cap_id_4, np.zeros([1,1])]).astype(int)
                    cap_id_5 = np.hstack([cap_id_5, np.zeros([1,1])]).astype(int)
                    ### caption labels ###
                    capl = np.zeros([cap_length])
                    capl[:n_words + 1] = 1
#                    print 'sample_data:', sample_data[0, 150: 170]
#                    print 'encode_data:', encode_data[43, 150: 170]
#                    print 'video_label:', vl[:]
#                    print 'caption_label:', capl[ :]
#                    print 'caption_id:', cap_id[ :]
#                    print 'caption_id_1:', cap_id_1[ :]
#                    print 'caption_id_2:', cap_id_2[ :]
#                    print 'caption_id_3:', cap_id_3[ :]
#                    print 'caption_id_4:', cap_id_4[ :]
#                    print 'caption_id_5:', cap_id_5[ :]
                    write_data_as_record(writer, sample_data, encode_data, video_name, title, vl, capl,
                        cap_id, cap_id_1, cap_id_2, cap_id_3, cap_id_4, cap_id_5, frame_data)
                    cnt += 1
    print 'totally', cnt, 'v2s pairs.'
    writer.close()

def trans_video_msrvtt_record(datasplit_list, datasplit, vgg_feat_name, c3d_feat_name, wordtoix):
#    assert datasplit in ['train', 'val', 'test']
    assert len(datasplit_list) > 0
    re = json.load(open('msrvtt2sent.json'))
    n_length = 45
    cap_length = 35

    initial = 0
    cnt = 0
    filename = feature_folder + datasplit + '.tfrecords'
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for ele in datasplit_list:
        if datasplit == 'train':
            vgg_feat_file = vgg_feat_folder_train + ele
            c3d_feat_file = c3d_feat_folder_train + ele
        elif datasplit == 'val':
            vgg_feat_file = vgg_feat_folder_val + ele
            c3d_feat_file = c3d_feat_folder_val + ele
        else:
            vgg_feat_file = vgg_feat_folder_test + ele
            c3d_feat_file = c3d_feat_folder_test + ele

        assert os.path.isfile(vgg_feat_file)
        assert os.path.isfile(c3d_feat_file)
        vgg_feat = np.squeeze(np.asarray(h5py.File(vgg_feat_file)[vgg_feat_name]))
        c3d_feat = np.squeeze(np.asarray(h5py.File(c3d_feat_file)[c3d_feat_name]))
        # to solve the number of frames mismatch between different videos
        sample_data = np.zeros([n_length, 4096 * 2])
        encode_data = np.zeros([n_length, 4096 * 2])
        concat_feat = np.concatenate((vgg_feat, c3d_feat), axis = 1)
        # post pad
        sample_data[:concat_feat.shape[0], :] = concat_feat
        # pre pad
        encode_data[-concat_feat.shape[0]:, :] = concat_feat
        video_name = ele
        if video_name in re.keys():
            print video_name, 'num_sen:', len(re[video_name])
            caps = re[video_name]
            for xxx in caps:
                en_cap_ind = random.sample(range(0, len(caps)), 5)
                if len(xxx.split(' ')) < 35:
                    title = unicodedata.normalize('NFKD', xxx).encode('ascii','ignore')
                    ### video label ###
                    vl = np.zeros([n_length])
                    vl[:concat_feat.shape[0]] = 1
                    ### caption of word ids ###
                    cap_id, n_words = get_cap_ids(xxx, wordtoix, cap_length, pad='post')
                    cap_id_1, n_word_1 = get_cap_ids(caps[en_cap_ind[0]], wordtoix, cap_length, pad='pre')
                    cap_id_2, n_word_2 = get_cap_ids(caps[en_cap_ind[1]], wordtoix, cap_length, pad='pre')
                    cap_id_3, n_word_3 = get_cap_ids(caps[en_cap_ind[2]], wordtoix, cap_length, pad='pre')
                    cap_id_4, n_word_4 = get_cap_ids(caps[en_cap_ind[3]], wordtoix, cap_length, pad='pre')
                    cap_id_5, n_word_5 = get_cap_ids(caps[en_cap_ind[4]], wordtoix, cap_length, pad='pre')
                    cap_id = np.hstack([cap_id, np.zeros([1,1])]).astype(int)
                    cap_id_1 = np.hstack([cap_id_1, np.zeros([1,1])]).astype(int)
                    cap_id_2 = np.hstack([cap_id_2, np.zeros([1,1])]).astype(int)
                    cap_id_3 = np.hstack([cap_id_3, np.zeros([1,1])]).astype(int)
                    cap_id_4 = np.hstack([cap_id_4, np.zeros([1,1])]).astype(int)
                    cap_id_5 = np.hstack([cap_id_5, np.zeros([1,1])]).astype(int)
                    ### caption labels ###
                    capl = np.zeros([cap_length])
                    capl[:n_words + 1] = 1
#                    print 'sample_data:', sample_data[0, 150: 170]
#                    print 'encode_data:', encode_data[43, 150: 170]
#                    print 'video_label:', vl[:]
#                    print 'caption_label:', capl[ :]
#                    print 'caption_id:', cap_id[ :]
#                    print 'caption_id_1:', cap_id_1[ :]
#                    print 'caption_id_2:', cap_id_2[ :]
#                    print 'caption_id_3:', cap_id_3[ :]
#                    print 'caption_id_4:', cap_id_4[ :]
#                    print 'caption_id_5:', cap_id_5[ :]
                    write_data_as_record(writer, sample_data, encode_data, video_name, title, vl, capl,
                        cap_id, cap_id_1, cap_id_2, cap_id_3, cap_id_4, cap_id_5)
                    cnt += 1
    print 'totally', cnt, 'v2s pairs.'
    writer.close()

def trans_video_youtube(datasplit_list, datasplit, vgg_feat_name,
        c3d_feat_name, wordtoix):
    assert len(datasplit_list) > 0
    re = json.load(open(v2s_json))
    batch_size = 100
    n_length = 45
    cap_length = 35

    initial = 0
    cnt = 0
    fname = []
    title = []
    data = []
    en_data = []
    video_label = []
    caption_label = []
    caption_id = []
    caption_id_1 = []
    caption_id_2 = []
    caption_id_3 = []
    caption_id_4 = []
    caption_id_5 = []
    for ele in datasplit_list:
        vgg_feat_file = vgg_feat_folder + ele + '.h5'
        c3d_feat_file = c3d_feat_folder + ele + '_c3d_' + c3d_feat_name + '-1.h5'
        assert os.path.isfile(vgg_feat_file)
        assert os.path.isfile(c3d_feat_file)
        vgg_feat = np.asarray(h5py.File(vgg_feat_file)[vgg_feat_name])
        c3d_feat = np.asarray(h5py.File(c3d_feat_file)['data'])
        # to solve the number of frames mismatch between different videos
        sample_data = np.zeros([n_length, 4096 * 2])
        encode_data = np.zeros([n_length, 4096 * 2])
        concat_feat = np.concatenate((vgg_feat, c3d_feat), axis = 1)
        # post pad
        sample_data[:concat_feat.shape[0], :] = concat_feat
        # pre pad
        encode_data[-concat_feat.shape[0]:, :] = concat_feat
        video_name = ele
        if video_name in re.keys():
            print video_name, 'num_sen:', len(re[video_name])
            caps = re[video_name]
            for xxx in caps:
                en_cap_ind = random.sample(range(0, len(caps)), 5)
                if len(xxx.split(' ')) < 35:
                    fname.append(video_name)
                    title.append(unicodedata.normalize('NFKD', xxx).encode('ascii','ignore'))
                    data.append(sample_data)
                    en_data.append(encode_data)
                    ### video label ###
                    vl = np.zeros([n_length])
                    vl[:concat_feat.shape[0]] = 1
                    video_label.append(vl)
                    ### caption of word ids ###
                    cap_id, n_words = get_cap_ids(xxx, wordtoix, cap_length, pad='post')
                    cap_id_1, n_word_1 = get_cap_ids(caps[en_cap_ind[0]], wordtoix, cap_length, pad='pre')
                    cap_id_2, n_word_2 = get_cap_ids(caps[en_cap_ind[1]], wordtoix, cap_length, pad='pre')
                    cap_id_3, n_word_3 = get_cap_ids(caps[en_cap_ind[2]], wordtoix, cap_length, pad='pre')
                    cap_id_4, n_word_4 = get_cap_ids(caps[en_cap_ind[3]], wordtoix, cap_length, pad='pre')
                    cap_id_5, n_word_5 = get_cap_ids(caps[en_cap_ind[4]], wordtoix, cap_length, pad='pre')
                    cap_id = np.hstack([cap_id, np.zeros([1,1])]).astype(int)
                    cap_id_1 = np.hstack([cap_id_1, np.zeros([1,1])]).astype(int)
                    cap_id_2 = np.hstack([cap_id_2, np.zeros([1,1])]).astype(int)
                    cap_id_3 = np.hstack([cap_id_3, np.zeros([1,1])]).astype(int)
                    cap_id_4 = np.hstack([cap_id_4, np.zeros([1,1])]).astype(int)
                    cap_id_5 = np.hstack([cap_id_5, np.zeros([1,1])]).astype(int)
                    caption_id.append(np.squeeze(cap_id))
                    caption_id_1.append(np.squeeze(cap_id_1))
                    caption_id_2.append(np.squeeze(cap_id_2))
                    caption_id_3.append(np.squeeze(cap_id_3))
                    caption_id_4.append(np.squeeze(cap_id_4))
                    caption_id_5.append(np.squeeze(cap_id_5))
                    ### caption labels ###
                    capl = np.zeros([cap_length])
                    capl[:n_words + 1] = 1
                    caption_label.append(capl)
                    cnt += 1
                    if cnt == batch_size:
                        batch = h5py.File(feature_folder + datasplit + '{:06d}'.format(initial) + '.h5','w')
                        batch['data'] = np.asarray(data) #np.zeros((batch_size, length, 4096*2))
                        batch['encode_data'] = np.asarray(en_data)
                        batch['fname'] = np.asarray(fname)
                        batch['title'] = np.asarray(title)
#                       batch['pos'] = np.zeros(batch_size)
                        batch['video_label'] = np.asarray(video_label) # np.zeros((batch_size, length))
                        batch['caption_label'] = np.asarray(caption_label)
                        batch['caption_id'] = np.asarray(caption_id)
                        batch['caption_id_1'] = np.asarray(caption_id_1)
                        batch['caption_id_2'] = np.asarray(caption_id_2)
                        batch['caption_id_3'] = np.asarray(caption_id_3)
                        batch['caption_id_4'] = np.asarray(caption_id_4)
                        batch['caption_id_5'] = np.asarray(caption_id_5)
                        fname = []
                        title = []
                        video_label = []
                        caption_label = []
                        caption_id = []
                        caption_id_1 = []
                        caption_id_2 = []
                        caption_id_3 = []
                        caption_id_4 = []
                        caption_id_5 = []
                        data = []
                        en_code = []
                        cnt = 0
                        initial += 1
        #### last batch ####
        if ele == datasplit_list[-1] and len(fname) > 0:
            while len(fname) < batch_size:
                fname.append('')
                title.append('')
            print 'data shape:', np.asarray(data).shape
            print 'fname shape: ', np.asarray(fname).shape
            print 'title shape: ', np.asarray(title).shape
            print 'caption_id shape:', np.asarray(caption_id).shape
            print 'caption_label shape: ', np.asarray(caption_label).shape
            print 'video_label shape:', np.asarray(video_label).shape
            batch = h5py.File(feature_folder + datasplit + '{:06d}'.format(initial) + '.h5','w')
            batch['data'] = np.zeros((batch_size, n_length, 4096*2))
            batch['data'][:len(data),:,:] = np.asarray(data)#np.zeros((batch_size,n_length, 4096*2))
            batch['encode_data'] = np.zeros((batch_size, n_length, 4096*2))
            batch['encode_data'][:len(en_data),:,:] = np.asarray(en_data)#np.zeros((batch_size,n_length, 4096*2))
            fname = np.array(fname)
            title = np.array(title)
            batch['fname'] = fname
            batch['title'] = title
            batch['video_label'] = np.zeros((batch_size, n_length))
            batch['video_label'][:len(data),:] = np.asarray(video_label) #np.zeros((batch_size, n_length))
            ### caption of word ids ###
            ci = np.zeros((batch_size, cap_length))
            ci[:len(caption_id), :] = np.array(caption_id) #np.zeros((batch_size, caption_length))
            batch['caption_id'] = ci
            ci_1 = np.zeros((batch_size, cap_length))
            ci_1[:len(caption_id_1), :] = np.array(caption_id_1) #np.zeros((batch_size, caption_length))
            batch['caption_id_1'] = ci
            ci_2 = np.zeros((batch_size, cap_length))
            ci_2[:len(caption_id_2), :] = np.array(caption_id_2) #np.zeros((batch_size, caption_length))
            batch['caption_id_2'] = ci_2
            ci_3 = np.zeros((batch_size, cap_length))
            ci_3[:len(caption_id_3), :] = np.array(caption_id_3) #np.zeros((batch_size, caption_length))
            batch['caption_id_3'] = ci_3
            ci_4 = np.zeros((batch_size, cap_length))
            ci_4[:len(caption_id_4), :] = np.array(caption_id_4) #np.zeros((batch_size, caption_length))
            batch['caption_id_4'] = ci_4
            ci_5 = np.zeros((batch_size, cap_length))
            ci_5[:len(caption_id_5), :] = np.array(caption_id_5) #np.zeros((batch_size, caption_length))
            batch['caption_id_5'] = ci_5
            ### caption labels ###
            capl = np.zeros((batch_size, cap_length))
            capl[:len(caption_label), :] = np.array(caption_label)#np.zeros((batch_size, caption_length))
            batch['caption_label'] = capl



def getlist(feature_folder_name, split):
    list_path = os.path.join(path, feature_folder_name+'/')
    List = glob.glob(list_path+split+'*.h5')
    f = open(list_path+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')


if __name__ == '__main__':
    dataset = np.load(home_folder + 'data0/msvd_dataset.npz')
    wordtoix, _ = build_vocab(dataset['train'], 'msvd')
#    trans_video_youtube_record(dataset['train'], 'train', vgg_feat_name, c3d_feat_name, wordtoix)
    trans_video_youtube_record(dataset['val'], 'val', vgg_feat_name, c3d_feat_name, wordtoix)
    trans_video_youtube_record(dataset['test'], 'test', vgg_feat_name, c3d_feat_name, wordtoix)
#    getlist(feature_folder,'train')
#    getlist(feature_folder,'val')
#    getlist(feature_folder,'test')

