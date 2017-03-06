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

def trans_video_youtube(datasplit_list, datasplit, vgg_feat_name,
        c3d_feat_name):
    assert len(datasplit_list) > 0
    re = json.load(open('msvd2sent.json'))
    batch_size = 100
    n_length = 45

    initial = 0
    cnt = 0
    fname = []
    title = []
    data = []
    label = []
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
                    ll = np.zeros([n_length]) - 1
                    ll[concat_feat.shape[0] - 1] = 0
                    label.append(ll)
                    cnt += 1
                    if cnt == batch_size:
                        batch = h5py.File(feature_folder + datasplit + '{:06d}'.format(initial) + '.h5','w')
                        data = np.array(data)
                        data = np.transpose(data,(1,0,2))
                        batch['data'] = data #np.zeros((n_length,batch_size,4096*2))
                        fname = np.array(fname)
                        title = np.array(title)
                        batch['fname'] = fname
                        batch['title'] = title
#                       batch['pos'] = np.zeros(batch_size)
                        batch['label'] = np.transpose(np.array(label)) # np.zeros((n_length,batch_size))
                        fname = []
                        title = []
                        label = []
                        data = []
                        cnt = 0
                        initial += 1
        #### last batch ####
        if ele == datasplit_list[-1] and len(fname) > 0:
            while len(fname) < batch_size:
                fname.append('')
                title.append('')
            batch = h5py.File(feature_folder + datasplit + '{:06d}'.format(initial) + '.h5','w')
            batch['data'] = np.zeros((n_length,batch_size,4096*2))
            batch['data'][:,:len(data),:] = np.transpose(np.array(data),(1,0,2))#np.zeros((n_length,batch_size,4096*2))
            fname = np.array(fname)
            title = np.array(title)
            batch['fname'] = fname
            batch['title'] = title
            batch['label'] = np.ones((n_length,batch_size))*(-1)
            batch['label'][:,:len(data)] = np.array(label).T #np.zeros((n_length,batch_size))



def getlist(feature_folder_name, split):
    list_path = os.path.join(path, feature_folder_name+'/')
    List = glob.glob(list_path+split+'*.h5')
    f = open(list_path+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')


if __name__ == '__main__':
    dataset = np.load('msvd_dataset.npz')
    trans_video_youtube(dataset['train'], 'train', vgg_feat_name, c3d_feat_name)
    trans_video_youtube(dataset['val'], 'val', vgg_feat_name, c3d_feat_name)
    trans_video_youtube(dataset['test'], 'test', vgg_feat_name, c3d_feat_name)
#    getlist(feature_folder,'train')
#    getlist(feature_folder,'val')
#    getlist(feature_folder,'test')

