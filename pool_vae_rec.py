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
from modules.variational_autoencoder import VAE

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

        self.vae = VAE(self.dim_hidden * 2, self.dim_hidden)

        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1),name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')
        self.decode_image_W = tf.Variable(tf.random_uniform([dim_hidden, dim_image], -0.1, 0.1, name='decode_image_W'))
        self.decode_image_b = tf.Variable(tf.random_uniform([dim_image]), name='decode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self, drop_sent='keep', drop_video='keep', caption_weight=1., video_weight=1., latent_weight=0.01):
        assert drop_sent in ['totally', 'random', 'keep']
        assert drop_video in ['totally', 'random', 'keep']
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_steps, self.dim_image]) # b x nv x d
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_steps]) # b x nv

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_steps]) # b x nc
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_steps]) # b x nc

        c2 = tf.zeros([self.batch_size, self.lstm2.state_size[1]]) # b x h
        m2 = tf.zeros([self.batch_size, self.lstm2.state_size[0]]) # b x h
        state2 = (c2, m2) # 2 x b x h

        ######## Encoding Stage #########
        # encoding video
        # mean pooling
        embed_video = tf.reduce_sum(video, axis=1) # b x d_im
        # embedding into (0, 1) range
        output1 = tf.nn.tanh(tf.nn.xw_plus_b(embed_video, self.encode_image_W, self.encode_image_b)) # b x h
        # encoding sentence
        with tf.variable_scope("model") as scope:
            for i in xrange(self.n_caption_steps):
                if i > 0: scope.reuse_variables()
                with tf.variable_scope("LSTM2"):
                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i]) # b x h
                    output2, state2 = self.lstm2_dropout(current_embed, state2) # b x h
        ######## Encoding Stage #########

        ######## Dropout Stage #########
        if drop_sent == 'totally':
            output2 = tf.constant(0) * output2
            output2 = tf.stop_gradient(output2)
        elif drop_sent == 'random':
            coeff = tf.floor(tf.random_uniform([1], 0, 1) + 0.5)
            output2 = coeff * output2
        if drop_video == 'totally':
            output1 = tf.constant(0) * output1
            output1 = tf.stop_gradient(output1)
        elif drop_video == 'random':
            coeff = tf.floor(tf.random_uniform([1], 0, 1) + 0.5)
            output1 = coeff * output1
        ######## Dropout Stage #########

        ######## Semantic Learning Stage ########
        input_state = tf.concat([output1, output2], 1) # b x (2 * h)
        loss_latent, output_semantic, z_mean, z_log_sigma_sq, z_sigma_sq, eps = self.vae(input_state)
        ######## Semantic Learning Stage ########

        ######## Decoding Stage ##########
        c3 = tf.zeros([self.batch_size, self.dim_hidden])
        h3 = tf.zeros([self.batch_size, self.dim_hidden])
        state3 = (c3, h3) # 2 x b x h
        current_embed = tf.zeros([self.batch_size, self.dim_hidden]) # b x h

        loss_caption = 0.0
        loss_video = 0.0

        ## decoding sentence without attention
        with tf.variable_scope("model") as scope:
            with tf.variable_scope("LSTM3"):
                _, state3 = self.lstm3_dropout(output_semantic, state3) # b x h
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

        ## decoding video without attention
        decode_image = tf.nn.xw_plus_b(output_semantic, self.decode_image_W, self.decode_image_b) # b x d_im
        euclid_loss = tf.reduce_sum(tf.square(tf.subtract(decode_image, embed_video))) # 1
        euclid_loss = euclid_loss / self.batch_size # 1
        loss_video += tf.reduce_sum(euclid_loss) # 1

        loss_caption = loss_caption / tf.reduce_sum(caption_mask)
        loss_video = loss_video / tf.reduce_sum(video_mask)

        loss = caption_weight * loss_caption + video_weight * loss_video + latent_weight * loss_latent
        return loss, loss_caption, loss_latent, loss_video, video, video_mask, caption, caption_mask, \
           z_mean, z_log_sigma_sq, z_sigma_sq, eps


    def build_sent_generator(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_steps])

        ####### Encoding Video ##########
        # encoding video
        embed_video = tf.reduce_sum(video, axis=1) # b x d_im
        # embedding into (0, 1) range
        output1 = tf.nn.tanh(tf.nn.xw_plus_b(embed_video, self.encode_image_W, self.encode_image_b)) # b x h
        ####### Encoding Video ##########

        ####### Semantic Mapping ########
        output2 = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        input_state = tf.concat([output1, output2], 1) # b x h, b x h
        _, output_semantic, _, _, _, _ = self.vae(input_state)
        ####### Semantic Mapping ########

        ####### Decoding ########
        c3 = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        h3 = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        state3 = (c3, h3) # n x 2 x h
        current_embed = tf.zeros([self.batch_size, self.dim_hidden]) # b x h

        generated_words = []

        with tf.variable_scope("model") as scope:
            scope.reuse_variables()
            with tf.variable_scope("LSTM3"):
                _, state3 = self.lstm3_dropout(output_semantic, state3) # b x h
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

        generated_words = tf.transpose(tf.stack(generated_words)) # n_caption_step x 1
        return video, video_mask, generated_words, lstm3_variables

    def build_video_generator(self):
        sent = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_steps])
        sent_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_steps])

        ####### Encoding Sentence ##########
        c2 = tf.zeros([self.batch_size, self.lstm2.state_size[1]]) # b x h
        m2 = tf.zeros([self.batch_size, self.lstm2.state_size[0]]) # b x h
        state2 = (c2, m2)
        with tf.variable_scope("model") as scope:
            for i in xrange(self.n_caption_steps):
                scope.reuse_variables()
                with tf.variable_scope("LSTM2"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, sent[:, i])
                    output2, state2 = self.lstm2_dropout(current_embed, state2) # b x h
        ####### Encoding Sentence ##########

        ####### Semantic Mapping ########
        output1 = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        input_state = tf.concat([output1, output2], 1) # b x (2 * h)
        _, output_semantic, _, _, _, _ = self.vae(input_state)
        ####### Semantic Mapping ########

        ####### Decoding ########
        decode_image = tf.nn.xw_plus_b(output_semantic, self.decode_image_W, self.decode_image_b)
        ####### Decoding ########

        return sent, sent_mask, decode_image

############### Global Parameters ###############
video_data_path_train = '/home/shenxu/data/msvd_feat_vgg_c3d_batch/train_vn.txt'
video_data_path_val = '/home/shenxu/data/msvd_feat_vgg_c3d_batch/val_vn.txt'
video_data_path_test = '/home/shenxu/data/msvd_feat_vgg_c3d_batch/test_vn.txt'
# seems to be no use
video_feat_path = '/disk_2T/shenxu/msvd_feat_vgg_c3d_batch/'

#model_path = '/Users/shenxu/Code/V2S-tensorflow/data0/models/'
model_path = '/home/shenxu/V2S-tensorflow/models/pool_vae/'
#test_data_folder = '/Users/shenxu/Code/V2S-tensorflow/data0/'
test_data_folder = '/home/shenxu/data/msvd_feat_vgg_c3d_batch/'

############## Train Parameters #################
dim_image = 4096*2
dim_hidden= 512*2
n_video_steps = 45
n_caption_steps = 35
n_epochs = 200
batch_size = 100
learning_rate = 0.0001 
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

def testing_one(sess, video_feat_path, ixtoword, video_tf, video_mask_tf, caption_tf, counter):
    pred_sent = []
    gt_sent = []
    IDs = []
    namelist = []
    #print video_feat_path
    test_data_batch = h5py.File(video_feat_path)
    gt_captions = json.load(open('msvd2sent.json'))

    video_feat = test_data_batch['data']
    video_mask = test_data_batch['video_label']

    generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})

    for ind in xrange(batch_size):
        cap_key = test_data_batch['fname'][ind]
        if cap_key == '':
            break
        else:
            generated_words = ixtoword[generated_word_index[ind]]
            punctuation = np.argmax(np.asarray(generated_words) == '.')+1
            generated_words = generated_words[:punctuation]
            #ipdb.set_trace()
            generated_sentence = ' '.join(generated_words)
            pred_sent.append([{'image_id':str(counter),'caption':generated_sentence}])
            namelist.append(cap_key)
            for i,s in enumerate(gt_captions[cap_key]):
                s = unicodedata.normalize('NFKD', s).encode('ascii','ignore')
                gt_sent.append([{'image_id':str(counter),'cap_id':i,'caption':s}])
                IDs.append(str(counter))
            counter += 1

    return pred_sent, gt_sent, IDs, counter, namelist

def testing_all(sess, test_data, ixtoword, video_tf, video_mask_tf, caption_tf):
    pred_sent = []
    gt_sent = []
    IDs_list = []
    flist = []
    counter = 0
    gt_dict = defaultdict(list)
    pred_dict = {}
    for _, video_feat_path in enumerate(test_data):
        [b,c,d, counter, fns] = testing_one(sess, video_feat_path, ixtoword, 
            video_tf, video_mask_tf, caption_tf, counter)
        pred_sent += b
        gt_sent += c
        IDs_list += d
        flist += fns

    for k,v in zip(IDs_list,gt_sent):
        gt_dict[k].append(v[0])

    new_flist = []
    new_IDs_list = []
    for k,v in zip(range(len(pred_sent)),pred_sent):
        if flist[k] not in new_flist:
            new_flist.append(flist[k])
            new_IDs_list.append(str(k))
            pred_dict[str(k)] = v

    return pred_sent, gt_sent, new_IDs_list, gt_dict, pred_dict

def test_all_videos(sess, test_data, sent_tf, sent_mask_tf, gen_video_tf):
    avg_loss = 0.
    for _, test_batch_path in enumerate(test_data):
        test_batch = h5py.File(test_batch_path)
        test_sent = test_batch['caption_id']
        sent_mask = test_batch['caption_label']
        pd_image = sess.run(gen_video_tf, feed_dict={sent_tf: test_sent, sent_mask_tf: sent_mask})
        gt_image = np.sum(test_batch['data'], axis=1) # b x d_im
        loss = np.sqrt(np.sum((pd_image - gt_image)**2, axis=1))
        avg_loss += np.sum(loss) / np.asarray(test_sent).shape[0]
    return avg_loss / len(test_data)

def train():
    assert os.path.isdir(model_path)
    print 'load meta data...'
    meta_data, train_data, val_data, test_data = \
        get_video_data_jukin(video_data_path_train, video_data_path_val, video_data_path_test)
    wordtoix = np.load('./data0/wordtoix.npy').tolist()
#    train_data = np.asarray([test_data_folder + 'train000000.h5', test_data_folder + 'train000001.h5'])
#    val_data = np.asarray([test_data_folder + 'train000002.h5'])
    print 'build model and session...'
    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_caption_steps=n_caption_steps,
            n_video_steps=n_video_steps,
            drop_out_rate = 0.5,
            bias_init_vector=None)

    ## GPU configurations
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.6)
    tf_loss, tf_loss_caption, tf_loss_rbm, tf_loss_video, \
        tf_video, tf_video_mask, tf_caption, tf_caption_mask, \
        tf_z_mean, tf_z_log_sigma_sq, tf_z_sigma_sq, tf_eps = model.build_model(drop_sent='random', video_weight=0., latent_weight=0.01)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,
        log_device_placement=False, gpu_options=gpu_options))
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
    ## initialize variables added for optimizer
    temp = set(tf.global_variables())
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))

    print 'train...'
    tStart_total = time.time()
    for epoch in range(n_epochs):
        index = np.arange(len(train_data))
        np.random.shuffle(index)
        train_data = train_data[index]

        tStart_epoch = time.time()
        loss_epoch = np.zeros(len(train_data))
        trained_batch = 0
        for current_batch_file_idx in xrange(len(train_data)):
            tStart = time.time()
            h5_batch = train_data[current_batch_file_idx]
            current_batch = h5py.File(h5_batch)
            current_feats = np.asarray(current_batch['data'])
            current_video_masks = np.asarray(current_batch['video_label'])
            current_caption_matrix = np.asarray(current_batch['caption_id'])
            current_caption_masks = np.asarray(current_batch['caption_label'])
            tEnd1 = time.time()
            _, loss_val, loss_caption, loss_latent, loss_video, z_mean, z_log_sigma_sq, z_sigma_sq, eps = sess.run(
                    [train_op, tf_loss, tf_loss_caption, tf_loss_rbm, tf_loss_video, tf_z_mean,
                    tf_z_log_sigma_sq, tf_z_sigma_sq, tf_eps],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })
            loss_epoch[current_batch_file_idx] = loss_val
            tStop = time.time()
            print "Epoch:", epoch, " Batch:", current_batch_file_idx, " Loss:", loss_val,
            print " Loss_caption:", loss_caption, " Loss_latent:", loss_latent, " Loss_video:", loss_video
            print "Time Cost:", round(tStop - tStart,2), "s"

        print "Epoch:", epoch, " done. Loss:", np.mean(loss_epoch)
        tStop_epoch = time.time()
        print "Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s"


        if np.mod(epoch, 1) == 0 or epoch == n_epochs - 1:
            print "Epoch ", epoch, " is done. Saving the model ..."
            with tf.device("/cpu:0"):
                saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

            ######### test sentence generation ##########
            current_batch = h5py.File(val_data[np.random.randint(0,len(val_data))])
            video_tf, video_mask_tf, caption_tf, lstm3_variables_tf = model.build_sent_generator()
            ixtoword = pd.Series(np.load('./data0/ixtoword.npy').tolist())
            [pred_sent, gt_sent, id_list, gt_dict, pred_dict] = testing_all(sess, train_data[-2:],
                ixtoword, video_tf, video_mask_tf, caption_tf)
            for key in pred_dict.keys():
                for ele in gt_dict[key]:
                    print "GT:  " + ele['caption']
                print "PD:  " + pred_dict[key][0]['caption']
                print '-------'
            [pred_sent, gt_sent, id_list, gt_dict, pred_dict] = testing_all(sess, val_data,
                ixtoword,video_tf, video_mask_tf, caption_tf)
            scorer = COCOScorer()
            total_score = scorer.score(gt_dict, pred_dict, id_list)

            ######### test video generation ##########
            sent_tf, sent_mask_tf, gen_images_tf = model.build_video_generator()
            loss = test_all_videos(sess, val_data, sent_tf, sent_mask_tf, gen_images_tf)
            print 'video generatioin MSE:', loss

        sys.stdout.flush()

    print "Finally, saving the model ..."
    with tf.device("/cpu:0"):
        saver.save(sess, os.path.join(model_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print "Total Time Cost:", round(tStop_total - tStart_total,2), "s"

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
        with tf.device('/gpu:'+str(args.gpu_id)):
            print 'using gpu:', args.gpu_id
            train()
    elif args.task == 'test':
        with tf.device('/gpu:'+str(args.gpu_id)):
            total_score = test(model_path = args.model)
