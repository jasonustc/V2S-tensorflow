import h5py
import numpy as np
import json
import pdb
import unicodedata
import glob
import os

feature_folder = '/disk_2T/shenxu/msvd_feat_vgg_c3d_batch/'
vgg_feat_folder = '/disk_2T/shenxu/msvd_feat_vgg/'
c3d_feat_folder = '/disk_2T/shenxu/msvd_feat_c3d/'
vgg_feat_name =  'fc6'
c3d_feat_name = 'fc6'
word_count_threshold = 1

def get_cap_ids(title, wordtoix):
    return map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], title)

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

def build_vocab(train_set):
    re = json.load(open('msvd2sent.json'))
    train_title = []
    for video in train_set:
        for title in re['video']:
            train_title.append(title)
    pdb.set_trace()
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(train_title, word_count_threshold)
    pdb.set_trace()
    np.save('wordtoix', wordtoix)
    np.save('ixtoword', ixtoword)
    np.save('bias_init_vector', bias_init_vector)
    return wordtoix, ixtoword



def trans_video_youtube(datasplit_list, datasplit, vgg_feat_name,
        c3d_feat_name, wordtoix):
    assert len(datasplit_list) > 0
    re = json.load(open('msvd2sent.json'))
    batch_size = 100
    n_length = 45
    cap_length = 35

    initial = 0
    cnt = 0
    fname = []
    title = []
    data = []
    video_label = []
    caption_label = []
    caption_id = []
    for ele in datasplit_list:
        vgg_feat_file = vgg_feat_folder + ele + '.h5'
        c3d_feat_file = c3d_feat_folder + ele + '_c3d_' + c3d_feat_name + '-1.h5'
        assert os.path.isfile(vgg_feat_file)
        assert os.path.isfile(c3d_feat_file)
        vgg_feat = np.array(h5py.File(vgg_feat_file)[vgg_feat_name])
        c3d_feat = np.array(h5py.File(c3d_feat_file)['data'])
        # to solve the number of frames mismatch between different videos
        sample_data = np.zeros([n_length, 4096 * 2])
        concat_feat = np.concatenate((vgg_feat, c3d_feat), axis = 1)
        sample_data[:concat_feat.shape[0], :] = concat_feat
        video_name = ele
        if video_name in re.keys():
            print video_name, 'num_sen:', len(re[video_name])
            for xxx in re[video_name]:
                #pdb.set_trace()
                if len(xxx.split(' ')) < 35:
                    fname.append(video_name)
                    title.append(unicodedata.normalize('NFKD', xxx).encode('ascii','ignore'))
                    data.append(sample_data)
                    ### video label ###
                    vl = np.zeros([n_length])
                    vl[:concat_feat.shape[0]] = 1
                    video_label.append(vl)
                    ### caption of word ids ###
                    ci = np.zeros([cap_length])
                    cap_id = get_cap_ids(xxx, wordtoix)
                    ci[:len(cap_id)] = np.array(cap_id)
                    caption_id.append(ci)
                    ### caption labels ###
                    capl = np.zeros([cap_length])
                    capl[:len(cap_ids)] = 1
                    caption_label.append(capl)
                    pdb.set_trace()
                    cnt += 1
                    if cnt == batch_size:
                        batch = h5py.File(feature_folder + datasplit + '{:06d}'.format(initial) + '.h5','w')
                        batch['data'] = np.array(data) #np.zeros((batch_size, length, 4096*2))
                        batch['fname'] = np.array(fname)
                        batch['title'] = np.array(title)
#                       batch['pos'] = np.zeros(batch_size)
                        batch['video_label'] = np.array(video_label) # np.zeros((batch_size, length))
                        batch['caption_label'] = np.array(caption_label)
                        batch['caption_id'] = np.array(caption_id)
                        fname = []
                        title = []
                        video_label = []
                        caption_label = []
                        caption_id = []
                        data = []
                        cnt = 0
                        initial += 1
        #### last batch ####
        if ele == datasplit_list[-1] and len(fname) > 0:
            while len(fname) < batch_size:
                fname.append('')
                title.append('')
            batch = h5py.File(feature_folder + datasplit + '{:06d}'.format(initial) + '.h5','w')
            batch['data'] = np.zeros((batch_size, n_length, 4096*2))
            batch['data'][:len(data),:,:] = np.array(data)#np.zeros((batch_size,n_length, 4096*2))
            batch['fname'] = np.array(fname)
            batch['title'] = np.array(title)
            batch['video_label'] = np.zeros((batch_size, n_length))
            batch['video_label'][:len(data),:] = np.array(label) #np.zeros((batch_size, n_length))
            ### caption of word ids ###
            ci = np.zeros((batch_size, cap_length))
            ci[:len(cap_id)] = np.array(caption_id)
            batch['caption_id'] = ci
            ### caption labels ###
            capl = np.zeros((batch_size, cap_length))
            capl[:len(cap_ids), :] = caption_label
            batch['caption_label'] = capl
            pdb.set_trace()



def getlist(feature_folder_name, split):
    list_path = os.path.join(path, feature_folder_name+'/')
    List = glob.glob(list_path+split+'*.h5')
    f = open(list_path+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')


if __name__ == '__main__':
    dataset = np.load('msvd_dataset.npz')
    wordtoix, _ = build_vocab(dataset['train'])
    trans_video_youtube(dataset['train'], 'train', vgg_feat_name, c3d_feat_name, wordtoix)
    trans_video_youtube(dataset['val'], 'val', vgg_feat_name, c3d_feat_name, wordtoix)
    trans_video_youtube(dataset['test'], 'test', vgg_feat_name, c3d_feat_name, wordtoix)
#    getlist(feature_folder,'train')
#    getlist(feature_folder,'val')
#    getlist(feature_folder,'test')

