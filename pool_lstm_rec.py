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
from modules.rbm import RBM
from utils.record_helper import read_and_decode

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
                        help='train or test', default='train', type=str)
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

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_caption_steps,
        n_video_steps, drop_out_rate, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_caption_steps = n_caption_steps
        self.drop_out_rate = drop_out_rate
        self.n_video_steps = n_video_steps

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        # encoding LSTM for sentence
        self.lstm2 = tf.contrib.rnn.LSTMCell(self.dim_hidden, use_peepholes=True, state_is_tuple=True)
        # decoding LSTM for sentence
        self.lstm3 = tf.contrib.rnn.LSTMCell(self.dim_hidden, use_peepholes=True, state_is_tuple=True)

        self.lstm2_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)
        self.lstm3_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm3,output_keep_prob=1 - self.drop_out_rate)

        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1),name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')
        self.decode_image_W = tf.Variable(tf.random_uniform([dim_hidden, dim_image], -0.1, 0.1, name='decode_image_W'))
        self.decode_image_b = tf.Variable(tf.random_uniform([dim_image]), name='decode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self, video, video_mask, caption, caption_mask):
        caption_mask = tf.cast(caption_mask, tf.float32)
        ######## Encoding Stage #########
        # encoding video
        # mean pooling
        pool_video = tf.reduce_sum(video, axis=1) # b x d_im
        embed_video = tf.nn.xw_plus_b(pool_video, self.encode_image_W, self.encode_image_b) # b x h
        ######## Encoding Stage #########


        ######## Decoding Stage ##########
        c3 = tf.zeros([self.batch_size, self.dim_hidden])
        h3 = tf.zeros([self.batch_size, self.dim_hidden])
        state3 = (c3, h3) # 2 x b x h
        current_embed = tf.zeros([self.batch_size, self.dim_hidden]) # b x h

        loss_caption = 0.0
        loss_video = 0.0

        ## decoding sentence without attention
        with tf.variable_scope("model") as scope:
            ## first input video feature
            with tf.variable_scope("LSTM3"):
                output3, state3 = self.lstm3_dropout(embed_video, state3)
            for i in xrange(n_caption_steps):
                scope.reuse_variables()
                with tf.variable_scope("LSTM3"):
                    output3, state3 = self.lstm3_dropout(current_embed, state3) # b x h
                labels = tf.expand_dims(caption[:,i], 1) # b x 1
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
                concated = tf.concat([indices, labels], 1) # b x 2
                onehot_labels = tf.sparse_to_dense(concated,
                    tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) # b x w
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i])
                logit_words = tf.nn.xw_plus_b(output3, self.embed_word_W, self.embed_word_b) # b x w
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit_words,
                    labels = onehot_labels) # b x 1
                cross_entropy = cross_entropy * caption_mask[:,i] # b x 1
                loss_caption += tf.reduce_sum(cross_entropy) # 1

        loss_caption = loss_caption / tf.reduce_sum(caption_mask)
        loss = loss_caption
        return loss, video, video_mask, caption, caption_mask


    def build_generator(self, video, video_mask):
        ####### Encoding Video ##########
        # encoding video
        pool_video = tf.reduce_sum(video, axis=1) # b x d_im
        # embedding
        embed_video = tf.nn.xw_plus_b(pool_video, self.encode_image_W, self.encode_image_b) # b x h
        ####### Encoding Video ##########

        ####### Decoding ########
        num_samples = tf.shape(video)[0]
        c3 = tf.zeros([num_samples, self.dim_hidden]) # b x h
        h3 = tf.zeros([num_samples, self.dim_hidden]) # b x h
        state3 = (c3, h3) # n x 2 x h
        current_embed = tf.zeros([num_samples, self.dim_hidden]) # b x h

        generated_words = []

        with tf.variable_scope("model") as scope:
            scope.reuse_variables()
            ### first input video
            with tf.variable_scope("LSTM3") as vs:
                output3, state3 = self.lstm3(embed_video, state3)
            for i in range(self.n_caption_steps):
                scope.reuse_variables()
                with tf.variable_scope("LSTM3") as vs:
                    output3, state3 = self.lstm3(current_embed, state3 ) # b x h
                    lstm3_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]
                logit_words = tf.nn.xw_plus_b(output3, self.embed_word_W, self.embed_word_b) # b x w
                max_prob_index = tf.argmax(logit_words, 1) # b
                generated_words.append(max_prob_index) # b
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
        ####### Decoding ########

        generated_words = tf.transpose(tf.stack(generated_words)) # b x n_caption_step
        return video, video_mask, generated_words, lstm3_variables

############### Global Parameters ###############
video_data_path_train = '/home/shenxu/data/msvd_feat_vgg_c3d_batch/train.tfrecords'
video_data_path_val = '/home/shenxu/data/msvd_feat_vgg_c3d_batch/val.tfrecords'
video_data_path_test = '/home/shenxu/data/msvd_feat_vgg_c3d_batch/test.tfrecords'
# seems to be no use
video_feat_path = '/disk_2T/shenxu/msvd_feat_vgg_c3d_batch/'

#model_path = '/Users/shenxu/Code/V2S-tensorflow/data0/models/'
model_path = '/home/shenxu/V2S-tensorflow/models/pool_lstm/'
test_data_folder = '/home/shenxu/data/msvd_feat_vgg_c3d_batch/'
home_folder = '/home/shenxu/V2S-tensorflow/'

############## Train Parameters #################
dim_image = 4096*2
dim_hidden= 512*2
n_video_steps = 45
n_caption_steps = 35
n_epochs = 200
batch_size = 100
learning_rate = 0.0001
prefetch = 10000
min_queue_examples = 1000
clip_norm = 35
n_train_samples = 49659
n_val_samples = 4149
n_test_samples = 27020
##################################################

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

    return pred_sent, gt_sent, new_IDs_list, gt_dict, pred_dict


def train():
    assert os.path.isdir(home_folder)
    assert os.path.isfile(video_data_path_train)
    assert os.path.isfile(video_data_path_val)
    print 'load meta data...'
    wordtoix = np.load(home_folder + 'data0/wordtoix.npy').tolist()
    print 'build model and session...'
    with tf.device("/gpu:0"):
        model = Video_Caption_Generator(
                dim_image=dim_image,
                n_words=len(wordtoix),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_caption_steps=n_caption_steps,
                n_video_steps=n_video_steps,
                drop_out_rate = 0.5,
                bias_init_vector=None)
    tStart_total = time.time()
    n_epoch_steps = int(n_train_samples / batch_size)
    n_steps = n_epochs * n_epoch_steps
    # build data reading graph on CPU
    with tf.device('/cpu:0'):
        train_data, train_encode_data, _, _, train_video_label, train_caption_label, train_caption_id, train_caption_id_1, \
            _, _, _, _ = read_and_decode(video_data_path_train)
        val_data, val_encode_data, val_fname, val_title, val_video_label, val_caption_label, val_caption_id, val_caption_id_1, \
            _, _, _, _ = read_and_decode(video_data_path_val)
        # random batches
        train_data, train_encode_data, train_video_label, train_caption_label, train_caption_id, train_caption_id_1 = \
            tf.train.shuffle_batch([train_data, train_encode_data, train_video_label, train_caption_label, train_caption_id, train_caption_id_1],
                batch_size=batch_size, num_threads=3, capacity=prefetch, min_after_dequeue=int(prefetch/4))
        val_data, val_video_label, val_fname = \
            tf.train.batch([val_data, val_video_label, val_fname], batch_size=batch_size, num_threads=3, capacity=1000)
    # build model on GPU
    with tf.device("/gpu:0"):
        tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask= model.build_model(train_data, train_video_label, train_caption_id, train_caption_label)
        val_video_tf, val_video_mask_tf, val_caption_tf, val_lstm3_variables_tf = model.build_generator(val_data, val_video_label)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    # check for model file
    with tf.device("/cpu:0"):
        saver = tf.train.Saver(max_to_keep=2)
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print_tensors_in_checkpoint_file(ckpt.model_checkpoint_path, "", True)
    else:
        print("Created model with fresh parameters.")
        sess.run(tf.global_variables_initializer())
    temp = set(tf.global_variables())
    # train on GPU
    with tf.device("/gpu:0"):
#        train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
        ## initialize variables added for optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gvs = optimizer.compute_gradients(tf_loss)
        # when variable is not related to the loss, grad returned as None
        clip_gvs = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in gvs if grad is not None]
        pdb.set_trace()
        train_op = optimizer.apply_gradients(gvs)

    sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))
    # initialize epoch variable in queue reader
    sess.run(tf.local_variables_initializer())
    loss_epoch = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for step in range(n_steps):
        tStart = time.time()
        _, loss_val = sess.run([train_op, tf_loss])
        tStop = time.time()
        print "step:", step, " Loss:", loss_val,
        print "Time Cost:", round(tStop - tStart, 2), "s"
        loss_epoch += loss_val

        if step % n_epoch_steps == 0:
            epoch = step / n_epoch_steps
            loss_epoch /= n_epoch_steps
            with tf.device("/cpu:0"):
                saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
            print 'epoch:', epoch, 'loss:', loss_epoch
            loss_epoch = 0
            ######### test sentence generation ##########
            ixtoword = pd.Series(np.load(home_folder + 'data0/ixtoword.npy').tolist())
            n_val_steps = int(n_val_samples / batch_size)
            [pred_sent, gt_sent, id_list, gt_dict, pred_dict] = testing_all(sess, 1, ixtoword, val_caption_tf, val_fname)
            for key in pred_dict.keys():
                for ele in gt_dict[key]:
                    print "GT:  " + ele['caption']
                print "PD:  " + pred_dict[key][0]['caption']
                print '-------'
            [pred_sent, gt_sent, id_list, gt_dict, pred_dict] = testing_all(sess, n_val_steps, ixtoword, val_caption_tf, val_fname)
            scorer = COCOScorer()
            total_score = scorer.score(gt_dict, pred_dict, id_list)
            sys.stdout.flush()

        sys.stdout.flush()

    coord.request_stop()
    coord.join(threads)
    print "Finally, saving the model ..."
    with tf.device("/cpu:0"):
        saver.save(sess, os.path.join(model_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print "Total Time Cost:", round(tStop_total - tStart_total,2), "s"
    sess.close()

def test(model_path='models/model-900', video_feat_path=video_feat_path):
    meta_data, train_data, val_data, test_data = get_video_data_jukin(video_data_path_train, video_data_path_val, video_data_path_test)
#    test_data = val_data   # to evaluate on testing data or validation data
    ixtoword = pd.Series(np.load('./data0/ixtoword.npy').tolist())

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate = 0,
            bias_init_vector=None)

    video_tf, video_mask_tf, caption_tf, lstm3_variables_tf = model.build_generator()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    with tf.device("/cpu:0"):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

    for ind, row in enumerate(lstm3_variables_tf):
        if ind % 4 == 0:
                assign_op = row.assign(tf.multiply(row,1-0.5))
                sess.run(assign_op)

    [pred_sent, gt_sent, id_list, gt_dict, pred_dict] = testing_all(sess, test_data, ixtoword,video_tf, video_mask_tf, caption_tf)
    #np.savez('Att_result/'+model_path.split('/')[1],gt = gt_sent,pred=pred_sent)
    scorer = COCOScorer()
    total_score = scorer.score(gt_dict, pred_dict, id_list)
    return total_score

if __name__ == '__main__':
    args = parse_args()
    if args.task == 'train':
        train()
    elif args.task == 'test':
        with tf.device('/gpu:'+str(args.gpu_id)):
            total_score = test(model_path = args.model)
