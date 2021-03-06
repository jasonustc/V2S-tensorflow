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
import re
from numpy import linalg as LA
import cv2, shutil, pickle

############### Global Parameters ###############
video_data_path_train = '/home/shenxu/data/msvd_feat_vgg_c3d_frame/train.tfrecords'
video_data_path_val = '/home/shenxu/data/msvd_feat_vgg_c3d_frame/val.tfrecords'
video_data_path_test = '/home/shenxu/data/msvd_feat_vgg_c3d_frame/test.tfrecords'
# seems to be no use
video_feat_path = '/disk_2T/shenxu/msvd_feat_vgg_c3d_batch/'

#model_path = '/Users/shenxu/Code/V2S-tensorflow/data0/models/'
test_data_folder = '/home/shenxu/data/msvd_feat_vgg_c3d_batch/'
home_folder = '/home/shenxu/V2S-tensorflow/'
wordtoix_file = home_folder + 'data0/msvd_wordtoix.npy'
ixtoword_file = home_folder + 'data0/msvd_ixtoword.npy'
bias_init_vector_file = home_folder + 'data0/msvd_bias_init_vector.npy'

############## Train Parameters #################
### MSVD ###
#dim_image = 6912
### MSRVTT ###
dim_image = 9216
dim_video_feat = 2*4096
dim_hidden = 512
dim_att = 1000
n_video_steps = 45
n_caption_steps = 35
n_epochs = 200
batch_size = 100
learning_rate = 0.001
num_threads = 3
min_queue_examples = batch_size
prefetch = num_threads * batch_size + min_queue_examples
clip_norm = 35
n_train_samples = 49659
n_val_samples = 4149
n_test_samples = 27020
feat_scale_factor = 0.013
pixel_scale_factor = 0.00392
#### MSVD ####
resize_height = 36
resize_width = 64
#### MSRVTT ####
#resize_height = 64
#resize_width = 48
##################################################

######### general operations #####################
def get_model_step(model_path):
#    assert os.path.isfile(model_path)
    MODEL_REGEX = r'model-(\d+)'
    model_file = os.path.basename(model_path)
    find = re.match(MODEL_REGEX, model_file)
    if find:
        return int(find.group(1))
    else:
        return 0

def get_global_step(folder):
    assert os.path.isdir(folder)
    model_list = os.listdir(folder)
    global_step = 0
    for model in model_list:
        step = get_model_step(model)
        if step > global_step:
            global_step = step
    return global_step

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Extract a CNN features')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--net', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset to extract',
                        default='train_val', type=str)
    parser.add_argument('--task', dest='task',
                        help='train or test',
                        default='train', type=str)
    parser.add_argument('--tg', dest='tg',
                        help='target to be extract lstm feature',
                        default='/home/Hao/tik/jukin/data/h5py', type=str)
    parser.add_argument('--ft', dest='ft',
                        help='choose which feature type would be extract',
                        default='lstm1', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def get_video_data(video_data_path, video_feat_path, train_ratio=0.9):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = video_data['video_path'].unique()
    train_len = int(len(unique_filenames)*train_ratio)

    train_vids = unique_filenames[:train_len]
    test_vids = unique_filenames[train_len:]

    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data

def get_video_data_HL(video_data_path, video_feat_path):
    files = open(video_data_path)
    List = []
    for ele in files:
        List.append(ele[:-1])
    return np.asarray(List)

def get_video_data_jukin(video_data_path_train, video_data_path_val, video_data_path_test):
    video_list_train = get_video_data_HL(video_data_path_train, video_feat_path)
    train_title = []
    title = []
    fname = []
    for ele in video_list_train:
        batch_data = h5py.File(ele)
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in xrange(len(batch_fname)):
                fname.append(batch_fname[i])
                title.append(batch_title[i])
                train_title.append(batch_title[i])

    video_list_val = get_video_data_HL(video_data_path_val, video_feat_path)
    for ele in video_list_val:
        batch_data = h5py.File(ele)
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in xrange(len(batch_fname)):
                fname.append(batch_fname[i])
                title.append(batch_title[i])

    video_list_test = get_video_data_HL(video_data_path_test, video_feat_path)
    for ele in video_list_test:
        batch_data = h5py.File(ele)
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in xrange(len(batch_fname)):
                fname.append(batch_fname[i])
                title.append(batch_title[i])

    video_data = pd.DataFrame({'Description':train_title})

    return video_data, video_list_train, video_list_val, video_list_test

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
    bias_init_vector = np.asarray([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector


def preProBuildLabel():
    ixtoword = {}
    wordtoix = {}
    ix = 1
    for w in range(1):
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return wordtoix, ixtoword
######### general operations #####################

######## testing related functions ###############

def testing_one(sess, ixtoword, caption_tf, fname_tf, counter):
    pred_sent = []
    gt_sent = []
    IDs = []
    namelist = []
    #print video_feat_path
    gt_captions = json.load(open(home_folder + 'msvd2sent.json'))

    generated_word_index, fname = sess.run([caption_tf, fname_tf])
    for ind in xrange(batch_size):
        cap_key = fname.values[ind]
        generated_words = ixtoword[generated_word_index[ind]]
        punctuation = np.argmax(np.asarray(generated_words) == '.')+1
        generated_words = generated_words[:punctuation]
        generated_sentence = ' '.join(generated_words)
        pred_sent.append([{'image_id':str(counter),'caption':generated_sentence}])
        namelist.append(cap_key)
        for i,s in enumerate(gt_captions[cap_key]):
            s = unicodedata.normalize('NFKD', s).encode('ascii','ignore')
            gt_sent.append([{'image_id':str(counter),'cap_id':i,'caption':s}])
            IDs.append(str(counter))
        counter += 1

    return pred_sent, gt_sent, IDs, counter, namelist

def testing_all(sess, n_steps, ixtoword, caption_tf, name_tf):
    pred_sent = []
    gt_sent = []
    IDs_list = []
    flist = []
    counter = 0
    gt_dict = defaultdict(list)
    pred_dict = {}
    for step in xrange(n_steps):
        [b,c,d, counter, fns] = testing_one(sess, ixtoword, caption_tf, name_tf, counter)
        pred_sent += b
        gt_sent += c
        IDs_list += d
        flist += fns

    for k,v in zip(IDs_list,gt_sent):
        gt_dict[k].append(v[0])

    new_flist = []
    new_IDs_list = []
    for k,v in zip(range(len(pred_sent)),pred_sent):
        # new video
        if flist[k] not in new_flist:
            new_flist.append(flist[k])
            new_IDs_list.append(str(k))
            pred_dict[str(k)] = v

    return pred_sent, gt_sent, new_IDs_list, gt_dict, pred_dict, new_flist

def get_demo_sentence(sess, n_steps, ixtoword, caption_tf, name_tf, result_file):
    [pred_sent, gt_sent, id_list, gt_dict, pred_dict, fname_list] = testing_all(sess, n_steps, ixtoword, caption_tf, name_tf)
    scorer = COCOScorer()
    scores = scorer.score(gt_dict, pred_dict, id_list, return_img_score=True)
    bleus = []
    for i, idx in enumerate(id_list):
        bleus.append((scores[idx]['Bleu_4'], fname_list[i], idx))
    sorted_bleus = sorted(bleus, key=lambda x: x[0], reverse=True)
    video_names = []
    with open(result_file, 'w') as result:
        for i in xrange(40):
            fname = sorted_bleus[i][1]
            video_names.append(fname)
            idx = sorted_bleus[i][2]
            result.write(fname + '\n')
            for ele in gt_dict[idx]:
                result.write('GT: ' + ele['caption'] + '\n')
            result.write('PD: ' + pred_dict[idx][0]['caption'] + '\n\n\n')
        print 'result saved to', result_file
    with open(result_file + '.videos', "wb") as fp:
        pickle.dump(video_names, fp)
        print 'video names saved to', result_file + '.videos'

def test_all_videos(sess, n_steps, gt_video_tf, gen_video_tf, video_label_tf, scale=None):
    avg_loss = 0.
    for ind in xrange(n_steps):
        loss = 0.
        gt_images, pd_images, video_label = sess.run([gt_video_tf, gen_video_tf, video_label_tf]) # b x n x d
        gt_images = gt_images[:, 1:, :]
        # recover from normalized feats
        if scale is not None:
            gt_images = scale * gt_images
        # only care about frames that counted
        pd_images = pd_images * np.expand_dims(video_label[:, 1:], 2)
        loss = np.sum((pd_images - gt_images)**2)
        avg_loss += loss / np.sum(video_label)
    return avg_loss / n_steps

def save_video_images(folder, pd_images, gt_images, video_label):
    for i in xrange(video_label.shape[0]):
        if video_label[i] == 1:
            cv2.imwrite(folder + '/pd_'+str(i)+'.jpg', np.reshape(pd_images[i]*255, (resize_height, resize_width, 3)))
            cv2.imwrite(folder + '/gt_'+str(i)+'.jpg', np.reshape(gt_images[i]*255, (resize_height, resize_width, 3)))
    print 'saved images to folder:', folder

def get_demo_video(sess, n_steps, gt_video_tf, gen_video_tf, video_label_tf, video_name_tf, save_folder, scale=None):
    if os.path.isdir(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)
    demo_video = {}
    min_loss = 10000.
    for i in xrange(n_steps):
        loss = 0.
        gt_images, pd_images, video_label, video_name = sess.run([gt_video_tf, gen_video_tf, video_label_tf, video_name_tf]) # b x n x d
        gt_images = gt_images[:, 1:, :]
        # recover from normalized feats
        if scale is not None:
            gt_images = scale * gt_images
        # only care about frames that counted
        pd_images = pd_images * np.expand_dims(video_label[:, 1:], 2)
        loss = np.sum((pd_images - gt_images)**2, axis=(1,2))
        label_sum = np.sum(video_label, axis=1)
        avg_loss = loss / label_sum
        ind = np.argmin(avg_loss)
#        if avg_loss[ind] < min_loss:
        if True:
            demo_video['name'] = video_name.values[ind]
            demo_video['pd_images'] = pd_images[ind, :, :]
            demo_video['gt_images'] = gt_images[ind, :, :]
            demo_video['video_label'] = video_label[ind, 1:]
            demo_video['loss'] = avg_loss[ind]
            min_loss = avg_loss[ind]
            if os.path.isdir(save_folder + demo_video['name']):
                shutil.rmtree(save_folder + demo_video['name'])
            os.mkdir(save_folder + demo_video['name'])
            print 'min_loss of batch', i, ':', avg_loss[ind]
            save_video_images(save_folder + demo_video['name'], demo_video['pd_images'], demo_video['gt_images'], demo_video['video_label'])
    return demo_video

######## testing related functions ###########o####
