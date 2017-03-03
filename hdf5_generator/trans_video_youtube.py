import h5py
import numpy as np
import json
import pdb
import unicodedata
import glob
import os

path = '/disk_new/shenxu/'
feature_folder = 'msvd_feat_vgg_batch/'
feat_folder = '/disk_new/shenxu/msvd_feat_vgg/'

def trans_video_youtube(datasplit_list, datasplit):
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
        feat_file = feat_folder + ele + '.h5'
        assert os.path.isfile(feat_file)
        print feat_file
        train_batch = h5py.File(feat_file)
        video_name = ele
        pdb.set_trace()
        #pdb.set_trace()
        if video_name in re.keys():
            for xxx in re[video_name]:
                #pdb.set_trace()
                if len(xxx.split(' ')) < 35:
                    pdb.set_trace()
                    fname.append(video_name)
                    print unicodedata.normalize('NFKD', xxx).encode('ascii','ignore')
                    title.append(unicodedata.normalize('NFKD', xxx).encode('ascii','ignore'))
                    data.append(train_batch['fc7'])
                    ll = np.zeros([n_length]) - 1
                    ll[len(train_batch['fc7'].shape[0] - 1)] = 0
                    label.append(ll)
                    cnt += 1
                    if cnt == batch_size:
                        batch = h5py.File(path+feature_folder+'/'+datasplit+str(initial)+'.h5','w')
                        data = np.transpose(data,(1,0,2))
                        batch['data'] = np.array(data)#np.zeros((n_length,batch_size,4096*2))
                        fname = np.array(fname)
                        title = np.array(title)
                        batch['fname'] = fname
                        batch['title'] = title
#                       batch['pos'] = np.zeros(batch_size)
                        batch['label'] = np.transpose(np.array(label))#np.zeros((n_length,batch_size))
                        fname = []
                        title = []
                        label = []
                        data = []
                        cnt = 0
                        initial += 1
        if ele == datasplit_list[-1] and len(fname) > 0:
            pdb.set_trace()
            while len(fname) < batch_size:
                fname.append('')
                title.append('')
            batch = h5py.File(path+feature_folder+'/'+datasplit+str(initial)+'.h5','w')
            batch['data'] = np.zeros((n_length,batch_size,4096))
            batch['data'][:,:len(data),:] = np.transpose(np.array(data),(1,0,2))#np.zeros((n_length,batch_size,4096))
            fname = np.array(fname)
            title = np.array(title)
            batch['fname'] = fname
            batch['title'] = title
            batch['label'] = np.ones((n_length,batch_size))*(-1)
            pdb.set_trace()
            batch['label'][:,:len(data)] = np.array(label).T
#            batch['pos'] = np.zeros(batch_size)
#            batch['label'] = np.zeros((n_length,batch_size))



def getlist(feature_folder_name, split):
    list_path = os.path.join(path, feature_folder_name+'/')
    List = glob.glob(list_path+split+'*.h5')
    f = open(list_path+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')


if __name__ == '__main__':
    dataset = np.load('msvd_dataset.npz')
    trans_video_youtube(dataset['train'], 'train')
    pdb.set_trace()
    trans_video_youtube(dataset['val'], 'val')
    pdb.set_trace()
    trans_video_youtube(dataset['test'], 'test')
    pdb.set_trace()
    getlist(feature_folder,'train')
    getlist(feature_folder,'val')
    getlist(feature_folder,'test')

