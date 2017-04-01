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

###### custom parameters #######
model_path = '/home/shenxu/V2S-tensorflow/models/att_vae/'
###### custom parameters #######

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
        # decoding LSTM for video
        self.lstm4 = tf.contrib.rnn.LSTMCell(self.dim_hidden, use_peepholes=True, state_is_tuple=True)

        self.lstm2_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)
        self.lstm3_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm3,output_keep_prob=1 - self.drop_out_rate)
        self.lstm4_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm4,output_keep_prob=1 - self.drop_out_rate)

        self.vae = VAE(self.dim_hidden * 2, self.dim_hidden)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')
        self.decode_image_W = tf.Variable(tf.random_uniform([dim_hidden, dim_image], -0.1, 0.1, name='decode_image_W'))
        self.decode_image_b = tf.Variable(tf.random_uniform([dim_image]), name='decode_image_b')
        self.embed_att_w = tf.Variable(tf.random_uniform([dim_hidden, 1], -0.1,0.1), name='embed_att_w')
        self.embed_att_Wa = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1,0.1), name='embed_att_Wa')
        self.embed_att_Ua = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden],-0.1,0.1), name='embed_att_Ua')
        self.embed_att_ba = tf.Variable( tf.zeros([dim_hidden]), name='embed_att_ba')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        self.embed_nn_Wp = tf.Variable(tf.random_uniform([3*dim_hidden, dim_hidden], -0.1,0.1), name='embed_nn_Wp')
        self.embed_nn_bp = tf.Variable(tf.zeros([dim_hidden]), name='embed_nn_bp')

    def build_model(self, video, video_mask, caption, caption_1, caption_mask, drop_sent='keep', 
        drop_video='keep', cap_weight=1., vid_weight=1., lat_weight=0.01):
        video_mask = tf.cast(video_mask, tf.float32)
        caption_mask = tf.cast(caption_mask, tf.float32)
        assert drop_sent in ['totally', 'random', 'keep']
        assert drop_video in ['totally', 'random', 'keep']

        video_flat = tf.reshape(video, [-1, self.dim_image]) # (b x nv) x d
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (b x nv) x h
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_steps, self.dim_hidden]) # b x nv x h
        image_emb = tf.transpose(image_emb, [1,0,2]) # n x b x h

        c_init = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        m_init = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        state2 = (c_init, m_init)

        ######## Encoding Stage #########
        # encoding video
        with tf.variable_scope("model") as scope:
            output1 = tf.reduce_mean(image_emb, axis = 0) # b x h 

        # encoding sentence
        with tf.variable_scope("model") as scope:
            for i in xrange(self.n_caption_steps):
                if i > 0: scope.reuse_variables()
                with tf.variable_scope("LSTM2"):
                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption_1[:,i])
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
        loss_latent, output_semantic = self.vae(input_state)
        ######## Semantic Learning Stage ########

        ######## Decoding Stage ##########
        state3 = (c_init, m_init)
        state4 = (c_init, m_init)
        video_prev = tf.zeros([self.batch_size, self.dim_hidden]) # b x d_im
        sent_prev = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        current_embed = tf.zeros([self.batch_size, self.dim_hidden]) # b x h

        loss_caption = 0.0
        loss_video = 0.0

        image_part = tf.reshape(image_emb, [-1, self.dim_hidden])
        image_part = tf.matmul(image_part, self.embed_att_Ua) + self.embed_att_ba
        image_part = tf.reshape(image_part, [self.n_video_steps, self.batch_size, self.dim_hidden])
        ## decoding sentence with attention
        with tf.variable_scope("model") as scope:
            with tf.variable_scope("LSTM3"):
                _, state3 = self.lstm3_dropout(tf.concat([output_semantic, m_init], 1), state3)
            for i in xrange(self.n_caption_steps):
                e = tf.tanh(tf.matmul(sent_prev, self.embed_att_Wa) + image_part) # n x b x h
                e = tf.reshape(e, [-1, self.dim_hidden])
                e = tf.matmul(e, self.embed_att_w) # n x b
                e = tf.reshape(e, [self.n_video_steps, self.batch_size])
    #            e = tf.reduce_sum(e,2) # n x b
                e_hat_exp = tf.multiply(tf.transpose(video_mask), tf.exp(e)) # n x b 
                denomin = tf.reduce_sum(e_hat_exp,0) # b
                denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
                alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h  # normalize to obtain alpha
                attention_list = tf.multiply(alphas, image_emb) # n x b x h
                atten = tf.reduce_sum(attention_list,0) # b x h       #  soft-attention weighted sum

                scope.reuse_variables()
                with tf.variable_scope("LSTM3"):
                    output3, state3 = self.lstm3_dropout(tf.concat([atten, current_embed], 1), state3) # b x h

                output3_2 = tf.tanh(tf.nn.xw_plus_b(tf.concat([output3,atten,current_embed], 1), 
                    self.embed_nn_Wp, self.embed_nn_bp)) # b x h
                sent_prev = output3 # b x h
                labels = tf.expand_dims(caption[:,i], 1) # b x 1
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
                concated = tf.concat([indices, labels], 1) # b x 2
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) # b x w
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i])

                logit_words = tf.nn.xw_plus_b(output3_2, self.embed_word_W, self.embed_word_b) # b x w
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit_words, 
                    labels = onehot_labels) # b x 1
                cross_entropy = cross_entropy * caption_mask[:,i] # b x 1
                loss_caption += tf.reduce_sum(cross_entropy) # 1

        ## decoding video with attention
        with tf.variable_scope("model") as scope:
            ## TODO: add attention for video decoding
            ## write into memory first
            with tf.variable_scope("LSTM4"):
                _, state4 = self.lstm4_dropout(output_semantic, state4)
            for i in xrange(self.n_video_steps):
                scope.reuse_variables()
                with tf.variable_scope("LSTM4"):
                    output4, state4 = self.lstm4_dropout(video_prev, state4)
                decode_image = tf.nn.xw_plus_b(output4, self.decode_image_W, self.decode_image_b) # b x d_im
                video_prev = image_emb[i, :, :] # b x h
                euclid_loss = tf.reduce_sum(tf.square(tf.subtract(decode_image, video[:,i,:])), 
                    axis=1, keep_dims=True) # b x 1 
                euclid_loss = euclid_loss * video_mask[:, i] # b x 1
                loss_video += tf.reduce_sum(euclid_loss) # 1

        loss_caption = loss_caption / tf.reduce_sum(caption_mask)
        loss_video = loss_video / tf.reduce_sum(video_mask)

        loss = cap_weight * loss_caption + lat_weight * loss_latent + vid_weight * loss_video
        return loss, loss_caption, loss_latent, loss_video, output_semantic


    def build_sent_generator(self, video, video_mask):
        video_mask = tf.cast(video_mask, tf.float32)

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_steps, self.dim_hidden])
        image_emb = tf.transpose(image_emb, [1,0,2]) # n x b x h

        ####### Encoding Video ##########
        c_init = tf.zeros([self.batch_size, self.dim_hidden]) # b x s0
        m_init = tf.zeros([self.batch_size, self.dim_hidden]) # b x s1
        output1 = tf.reduce_mean(image_emb, axis=0) # b x h
        ####### Encoding Video ##########

        ####### Semantic Mapping ########
        output2 = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        input_state = tf.concat([output1, output2], 1) # b x (2 * h)
        _, output_semantic = self.vae(input_state)
        ####### Semantic Mapping ########

        ####### Decoding ########
        state3 = (c_init, m_init) # n x 2 x h
        h_prev = tf.zeros([self.batch_size, self.dim_hidden])

        generated_words = []

        current_embed = tf.zeros([self.batch_size, self.dim_hidden])
        image_part = tf.reshape(image_emb, [-1, self.dim_hidden])
        image_part = tf.matmul(image_part, self.embed_att_Ua) + self.embed_att_ba
        image_part = tf.reshape(image_part, [self.n_video_steps, self.batch_size, self.dim_hidden])
        with tf.variable_scope("model") as scope:
            scope.reuse_variables()
            with tf.variable_scope("LSTM3"):
                _, state3 = self.lstm3(tf.concat([output_semantic, m_init], 1), state3)
            for i in range(self.n_caption_steps):
                e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part) # n x b x h
                e = tf.reshape(e, [-1, self.dim_hidden])
                e = tf.matmul(e, self.embed_att_w) # n x b
                e = tf.reshape(e, [self.n_video_steps, self.batch_size])
    #            e = tf.reduce_sum(e,2) # n x b
                e_hat_exp = tf.multiply(tf.transpose(video_mask), tf.exp(e)) # n x b
                denomin = tf.reduce_sum(e_hat_exp,0) # b
                denomin = denomin + tf.to_float(tf.equal(denomin, 0))
                alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h
                attention_list = tf.multiply(alphas, image_emb) # n x b x h
                atten = tf.reduce_sum(attention_list,axis=0) # b x h

                with tf.variable_scope("LSTM3") as vs:
                    output3, state3 = self.lstm3( tf.concat([atten, current_embed], 1), state3 ) # b x h
                    lstm3_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]

                output3_2 = tf.tanh(tf.nn.xw_plus_b(tf.concat([output3,atten,current_embed], 1), 
                    self.embed_nn_Wp, self.embed_nn_bp)) # b x h
                h_prev = output3
                logit_words = tf.nn.xw_plus_b(output3_2, self.embed_word_W, self.embed_word_b) # b x w
                max_prob_index = tf.argmax(logit_words, 1) # b
                generated_words.append(max_prob_index) # b
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
        ####### Decoding ########

        generated_words = tf.transpose(tf.stack(generated_words)) # n_caption_step x 1
        return generated_words, lstm3_variables

    def build_video_generator(self, sent):
        ####### Encoding Sentence ##########
        c_init = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        m_init = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        state2 = (c_init, m_init)
        with tf.variable_scope("model") as scope:
            scope.reuse_variables()
            # encoding sentence
            for i in xrange(self.n_caption_steps):
                with tf.variable_scope("LSTM2"):
                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, sent[:, i])
                    output2, state2 = self.lstm2_dropout(current_embed, state2) # b x h
        ####### Encoding Sentence ##########

        ####### Semantic Mapping ########
        output1 = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        input_state = tf.concat([output1, output2], axis=1) # b x (2 * h)
        _, output_semantic = self.vae(input_state)
        ####### Semantic Mapping ########

        ####### Decoding ########
        state4 = (c_init, m_init) # n x 2 x h
        image_emb = tf.zeros([self.batch_size, self.dim_hidden])

        generated_images = []

        with tf.variable_scope("model") as scope:
            scope.reuse_variables()
            with tf.variable_scope("LSTM4"):
                _, state4 = self.lstm4(output_semantic, state4)
            for i in range(self.n_video_steps):
                with tf.variable_scope("LSTM4") as vs:
                    output4, state4 = self.lstm4(image_emb, state4) # b x h
                    lstm4_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]

                image_prev = tf.nn.xw_plus_b(output4, self.decode_image_W, self.decode_image_b)
                image_emb = tf.nn.xw_plus_b(image_prev, self.encode_image_W, self.encode_image_b)
                generated_images.append(image_prev) # b x d_im
        ####### Decoding ########

        generated_images = tf.transpose(tf.stack(generated_images), [1, 0, 2]) # b x n_video_step x d_im
        return generated_images, lstm4_variables

def train():
    assert os.path.isdir(home_folder)
    assert os.path.isfile(video_data_path_train)
    assert os.path.isfile(video_data_path_val)
    assert os.path.isdir(model_path)
    print 'load meta data...'
    wordtoix = np.load(home_folder + 'data0/wordtoix.npy').tolist()
    print 'build model and session...'
    # place shared parameters on the GPU
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
    # preprocessing on the CPU
    with tf.device('/cpu:0'):
        train_data, train_encode_data, _, _, train_video_label, train_caption_label, train_caption_id, train_caption_id_1, \
            _, _, _, _ = read_and_decode(video_data_path_train)
        val_data, val_encode_data, val_fname, val_title, val_video_label, val_caption_label, val_caption_id, val_caption_id_1, \
            _, _, _, _ = read_and_decode(video_data_path_val)
        # random batches
        train_data, train_encode_data, train_video_label, train_caption_label, train_caption_id, train_caption_id_1 = \
            tf.train.shuffle_batch([train_data, train_encode_data, train_video_label, train_caption_label, train_caption_id, train_caption_id_1],
                batch_size=batch_size, num_threads=num_threads, capacity=prefetch, min_after_dequeue=min_queue_examples)
        val_data, val_encode_data, val_video_label, val_fname, val_caption_id, val_caption_id_1 = \
            tf.train.batch([val_data, val_encode_data, val_video_label, val_fname, val_caption_id, val_caption_id_1], batch_size=batch_size, num_threads=1, capacity=2*batch_size)
    # operation on the GPU
    with tf.device("/gpu:0"):
        tf_loss, tf_loss_caption, tf_loss_latent, tf_loss_video, tf_output_semantic= model.build_model(train_data, train_encode_data, train_caption_id, train_caption_id_1, train_caption_label)
        val_caption_tf, val_lstm3_variables_tf = model.build_sent_generator(val_data, val_video_label)
        val_video_tf, val_lstm4_variables_tf = model.build_video_generator(val_caption_id_1)

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    # check for model file
    with tf.device("/cpu:0"):
        saver = tf.train.Saver(max_to_keep=100)
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print_tensors_in_checkpoint_file(ckpt.model_checkpoint_path, "", True)
    else:
        print("Created model with fresh parameters.")
        sess.run(tf.global_variables_initializer())
    temp = set(tf.global_variables())
    # train on the GPU
    with tf.device("/gpu:0"):
#        train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gvs = optimizer.compute_gradients(tf_loss)
        # when variable is not related to the loss, grad returned as None
        clip_gvs = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in gvs if grad is not None]
        train_op = optimizer.apply_gradients(gvs)

    ## initialize variables added for optimizer
    sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))
    # initialize epoch variable in queue reader
    sess.run(tf.local_variables_initializer())
    loss_epoch = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # write graph architecture to file
    summary_writer = tf.summary.FileWriter(model_path + 'summary', sess.graph)
    for step in xrange(1, n_steps+1):
        tStart = time.time()
        _, loss_val, loss_cap, loss_lat, loss_vid, sem = sess.run([train_op, tf_loss, tf_loss_caption, tf_loss_latent, tf_loss_video, tf_output_semantic])
        tStop = time.time()
        print "step:", step, " Loss:", loss_val,
        print "Time Cost:", round(tStop - tStart, 2), "s"
        loss_epoch += loss_val

        if step % n_epoch_steps == 0:
            epoch = step / n_epoch_steps
            loss_epoch /= n_epoch_steps
            with tf.device("/cpu:0"):
                saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
            print 'epoch:', epoch, 'loss:', loss_epoch, 'loss_cap:', loss_cap, 'loss_lat:', loss_lat, 'loss_vid:', loss_vid
            print 'sem:', sem[0, :10]
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
            ######### test video generation #############
            mse = test_all_videos(sess, n_val_steps, val_data, val_video_tf)
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
    ixtoword = pd.Series(np.load(home_folder + 'data0/ixtoword.npy').tolist())

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
