from scipy.spatial.distance import pdist ,squareform
from scipy.stats import norm
import random
import tqdm
import scipy
import h5py
import scipy.io as sio
import tensorflow as tf
import numpy as np
from ops import *
from utils import *
import os, sys
import batch_data.image_data as dataset
from my_generator import Vgg

def loss(batch_input,batch_label):
    pair_loss=tf.reduce_mean(tf.multiply(tf.abs(batch_label),(tf.square(tf.multiply(1.0/hidden_size,tf.matmul(batch_input, tf.transpose(batch_input)))- batch_label))))
    return pair_loss

def inference(x224):
    with tf.variable_scope("enc"):
            vgg_net = Vgg('./vgg.npy', codelen=hidden_size)
            vgg_net.build(x224, train_model)
            z_x = vgg_net.fc9
            fc7_features = vgg_net.relu7
    return z_x, fc7_features

# Define the deep model and optimization method
batch_size = 24 
hidden_size = 32
input_image = tf.placeholder(tf.float32, [None, 224 ,224,3])
train_model = tf.placeholder(tf.bool)
input_label = tf.placeholder(tf.float32, [batch_size, batch_size])
with tf.device('/gpu:0'):
    z_x, fc_features = inference(input_image)
    pair_loss = loss(z_x, input_label)
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

# Define training dataset
config = {
    'img_tr': "dataset/flickr/img_train.txt", 
    'txt_tr': "dataset/flickr/label_train.txt", 
    'lab_tr': "dataset/flickr/label_train.txt",
    'img_te': "dataset/flickr/img_test.txt",
    'txt_te': "dataset/flickr/label_test.txt",
    'lab_te': "dataset/flickr/label_test.txt",
    'img_db': "dataset/flickr/img_database.txt", 
    'txt_db': "dataset/flickr/label_database.txt", 
    'lab_db': "dataset/flickr/label_database.txt",
    'n_train': 10000,
    'n_test': 2000,
    'n_db': 18015,
    'n_label': 24
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
globa_beta_indx = 0
num_examples = n_train
total_batch = int(np.floor(num_examples / batch_size))
epoch = 0     

# Constructing Semantic Similarity Matrix
# Extract Deep features
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
    train_softcode = session.run(fc_features, feed_dict = {input_image: next_batches224, train_model: False})
    train_features[index, :] = train_softcode
    train_labels[index, :] = batch_label
_dict = {'train_features': train_features, 'train_labels': train_labels}
np.save('train_feature_and_label_nuswide.npy', _dict)

# Calculate cosine distance 
euc_ = pdist(train_features, 'cosine')
euc_dis = squareform(euc_)
orig_euc_dis = euc_dis
start = -0.00000001
margin = 1.0/100
num = np.zeros(100)
max_num = 0.0
max_value = 0.0

# Histogram distribution 
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

# Gaussian distribution approximation
l_mean, l_std = norm.fit(left_all)
r_mean, r_std = norm.fit(right_all)

# Obtain fake labels
S1 = ((orig_euc_dis < l_mean-2*l_std))*1.0
S2 = ((orig_euc_dis > r_mean+2*r_std))*(-1.0)
S = S1 + S2

# Start training/
while epoch < pre_epochs:
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
                input_image: next_batches224,
                input_label: ss_,
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
            test_softcode = session.run(z_x, feed_dict = {input_image: next_batches224, train_model: False})
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
            dataset_softcode = session.run(z_x, feed_dict = {input_image: next_batches224, train_model: False})
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
