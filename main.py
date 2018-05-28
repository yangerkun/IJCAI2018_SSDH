from scipy.spatial.distance import pdist ,squareform
from scipy.stats import norm
import random
import tqdm
import prettytensor as pt
import scipy
import h5py
import scipy.io as sio
import tensorflow as tf
import numpy as np
from ops import *
from utils import *
import os, sys
sys.path.append(os.getcwd())
import batch_data.mxnet_image_data as dataset
from my_generator import Vgg
import numpy as np
import sklearn.datasets
import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')

def calc_map_k(qB, rB, query_L, retrieval_L, k):
    num_query = query_L.shape[0]
    map = 0
    for iter in xrange(num_query):
        gnd2 = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        aaa = np.arange(0,retrieval_L.shape[0])
        ind = np.lexsort((aaa,hamm))
        gnd = gnd2[ind[0:k]]
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map

def calc_map(qB, rB, query_L, retrieval_L):
    num_query = query_L.shape[0]
    map = 0
    for iter in xrange(num_query):
        gnd2 = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        aaa = np.arange(0,retrieval_L.shape[0])
        ind = np.lexsort((aaa,hamm))
        gnd = gnd2[ind[:]]
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map

def calc_hammingDist(B1, B2):
    B1 = B1*1
    B2 = B2*1
    ind = B1<0.5
    B1[ind] = -1
    ind = B2<0.5
    B2[ind] = -1
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1,B2.transpose()))
    return distH

def one_hot_label(single_label):
    num_label = np.max(single_label)+1
    num_samples = single_label.size
    one_hot_label = np.zeros([num_samples, num_label], int)
    for i in tqdm.tqdm(range(num_samples)):
        one_hot_label[i, single_label[i][0]] = 1
    return one_hot_label

def data_iterator():
    while True:
        idxs = np.arange(0, len(img224))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(img224), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = img224[cur_idxs]
            if len(images_batch) < batch_size:
                break
            images_batch = images_batch.astype("float32")
            yield images_batch ,cur_idxs

def test_iterator():
    while True:
        idxs = np.arange(0, len(test_img224))
        #np.random.shuffle(idxs)
        for batch_idx in range(0, len(test_img224), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = test_img224[cur_idxs]
            if len(images_batch) < batch_size:
                break
            images_batch = images_batch.astype("float32")
            yield images_batch ,cur_idxs

def dataset_iterator():
    while True:
        idxs = np.arange(0, len(dataset_img224))
        #np.random.shuffle(idxs)
        for batch_idx in range(0, len(dataset_img224), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = dataset_img224[cur_idxs]
            if len(images_batch) < batch_size:
                break
            images_batch = images_batch.astype("float32")
            yield images_batch ,cur_idxs

def loss(z_x_meanx1,ss_ ):
    pair_loss=tf.reduce_mean(tf.multiply(tf.abs(ss_),(tf.square(tf.multiply(1.0/hidden_size,tf.matmul(z_x_meanx1, tf.transpose(z_x_meanx1)))- ss_))))
    return pair_loss

def inference(x224):
    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with tf.variable_scope("enc"):
                vgg_net = Vgg('./vgg.npy', codelen=hidden_size)
                vgg_net.build(x224, train_model)
                z_x = vgg_net.fc9
                fc7_features = vgg_net.relu7
        return z_x, fc7_features

batch_size = 24 
hidden_size = 32
all_input224 = tf.placeholder(tf.float32, [None, 224 ,224,3])
train_model = tf.placeholder(tf.bool)
s_s = tf.placeholder(tf.float32, [batch_size, batch_size])
with tf.device('/gpu:0'):
    z_x, fc7_features = inference(all_input224)
    pair_loss = loss(z_x, s_s)
    params = tf.trainable_variables()
    E_params = [i for i in params if 'enc' in i.name]
    lr_E = tf.placeholder(tf.float32, shape=[])
    opt_E = tf.train.AdamOptimizer(lr_E, epsilon=1.0)            
    grads_e = opt_E.compute_gradients(pair_loss, var_list=E_params)#with KL_loss,you can discard it.

global_step = tf.get_variable(
    'global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)
train_E = opt_E.apply_gradients(grads_e, global_step=global_step)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
session = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

train_size = 5000
test_size = 10000
db_size = 50000
k = 50
config = {
    
    'img_tr': "cifar10/img_train.txt", 
    'lab_tr': "cifar10/label_train.txt",
    'img_te': "cifar10/img_test.txt",
    'lab_te': "cifar10/label_test.txt",
    'img_db': "cifar10/img_database.txt", 
    'lab_db': "cifar10/label_database.txt",
    'n_train': 5000,
    'n_test': 10000,
    'n_db': 50000,
    'n_label': 10
}

n_train = config['n_train']
n_test = config['n_test']
n_db = config['n_db']
n_label = config['n_label']
train_size = n_train
test_size = n_test
db_size = n_db

train_data = dataset.import_train(config)
train_labels = np.zeros([n_train, n_label])
train_features = np.zeros([ n_train, 4096])
test_data = dataset.import_test(config)
test_labels = np.zeros([n_test, n_label])
db_data = dataset.import_db(config)
db_labels = np.zeros([n_db, n_label])

num_epochs =  100
e_learning_rate = 1e-3
g_learning_rate = 1e-3
d_learning_rate = 1e-3
globa_beta_indx = 0
num_examples = n_train
total_batch = int(np.floor(num_examples / batch_size))
epoch = 0     

# Constructing Semantic Similarity Matrix
session.run(tf.initialize_all_variables())
pre_epochs =  10
train_batch = int(np.ceil(1.0*n_train/batch_size))
for i in range(train_batch):
    if (i+1)*batch_size < n_train:
        index = range(i*batch_size,(i+1)*batch_size)
    else:
        index = range(i*batch_size, n_train)
    next_batches224, batch_label = train_data.img_data(index)
    next_batches224 = np.array(next_batches224)
    train_softcode = session.run(fc7_features, feed_dict = {all_input224: next_batches224, train_model: False})
    train_features[index, :] = train_softcode
    train_labels[index, :] = batch_label
_dict = {'train_features': train_features, 'train_labels': train_labels}
np.save('train_feature_and_label_nuswide.npy', _dict)
euc_ = pdist(train_features, 'cosine')
euc_dis = squareform(euc_)
orig_euc_dis = euc_dis
start = -0.00000001
margin = 1.0/100
num = np.zeros(100)
max_num = 0.0
max_value = 0.0
for i in range(100):
    end = start+margin
    temp_matrix = (euc_dis>start)&(euc_dis<end)
    num[i] = np.sum(temp_matrix)
    if num[i]>max_num:
        max_num = num[i]
        max_value = start
    start = end
euc_dis = euc_dis.reshape(-1,1)
left = []
right = []
for i in range(euc_dis.shape[0]):
    if euc_dis[i] <= max_value:
        left.append(euc_dis[i])
    else:
        right.append(euc_dis[i])
left = np.array(left)
right = np.array(right)
fake_right = 2*max_value - left 
fake_left = 2*max_value - right 
left_all = np.concatenate([left, fake_right])
right_all = np.concatenate([fake_left, right])
l_mean, l_std = norm.fit(left_all)
S1 = ((orig_euc_dis < l_mean-2*l_std))*1.0
S2 = ((orig_euc_dis > l_mean-l_std))*(-1.0)
S = S1 + S2

# Start training/
while epoch < pre_epochs:
    iter_ = data_iterator()
    index_range = np.arange(n_train)
    np.random.shuffle(index_range)
    for i in  range(total_batch):
        if (i+1)*batch_size < n_train:
            index = index_range[range(i*batch_size,(i+1)*batch_size)]
        else:
            index = index_range[range(n_train - batch_size, n_train)]
        e_current_lr = e_learning_rate*1.0
        next_batches224,batch_label = train_data.img_data(index)
        next_batches224 = np.array(next_batches224)
        ss_ = S[index,:][:,index]
        _, PP_err= session.run(
            [
             train_E, pair_loss
             ],
            {
                lr_E: e_current_lr,
                all_input224: next_batches224,
                s_s: ss_,
                train_model: True
            }
            )
        print 'epoch:{}, batch: {},PP_err:{}'.format(epoch,i, PP_err)
    epoch = epoch + 1
    # Test for every 2 epoches.
    if (epoch+1) % 2 ==0 :
        test_codes = np.zeros([n_test,hidden_size])
        test_labels = np.zeros([n_test,n_label])
        dataset_codes = np.zeros([n_db,hidden_size])
        dataset_labels = np.zeros([n_db, n_label])
        test_batch = int(np.ceil(1.0*test_size/batch_size))
        dataset_batch =int(np.ceil(1.0*db_size/batch_size))
        #Extract hash codes for test dataset
        for i in range(test_batch):
            if (i+1)*batch_size < n_test:
                index = range(i*batch_size,(i+1)*batch_size)
            else:
                index = range(i*batch_size, n_test)
            next_batches224, batch_label = test_data.img_data(index)
            next_batches224 = np.array(next_batches224)
            test_softcode = session.run(z_x, feed_dict = {all_input224: next_batches224, train_model: False})
            test_codes[index, :] = test_softcode
            test_labels[index,:] = batch_label
        #Extract hash codes for database dataset
        for i in range(dataset_batch):
            if (i+1)*batch_size < n_db:
                index = range(i*batch_size,(i+1)*batch_size)
            else:
                index = range(i*batch_size, n_db)
            next_batches224, batch_label = db_data.img_data(index)
            next_batches224 = np.array(next_batches224)
            dataset_softcode = session.run(z_x, feed_dict = {all_input224: next_batches224, train_model: False})
            dataset_codes[index, :] = dataset_softcode
            dataset_labels[index, :] = batch_label
        # Caculate MAP values.
        dataset_codes = (dataset_codes>0)*1
        test_codes = (test_codes>0)*1
        dataset_L = dataset_labels  
        test_L = test_labels 
        dict_ = {'dataset_codes':dataset_codes, 'test_codes': test_codes, 'dataset_L': dataset_L, 'test_L': test_L}
        map_1000 = calc_map_k(test_codes, dataset_codes, test_L, dataset_L, 1000)
        map_ = calc_map(test_codes, dataset_codes, test_L, dataset_L)
        print 'pre: epoch:{}, map_1000:{}, map:{}'.format(epoch, map_1000, map_)
        np.save('./result/cifar10/32bit/'+str(epoch) +'.npy', dict_)
