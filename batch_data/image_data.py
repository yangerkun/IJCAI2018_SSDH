from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from six.moves import xrange  
from scipy.misc import imread, imresize
import cv2
import numpy as np
import h5py

class dataset(object):
    def __init__(self, i_path, l_path, train=True):
        self.i_lines = open(i_path, 'r').readlines()
        self.l_lines = open(l_path, 'r').readlines()
        print ('image numbers:{}',len(self.i_lines))
        print ('label numbers:{}',len(self.l_lines))
        self.n_samples = len(self.i_lines)
        self.train = train
        self._img  = [0] * self.n_samples
        self._label = [0] * self.n_samples
        self._load = [0] * self.n_samples
        self._txt  = [0] * self.n_samples
        self._load_num = 0            
        self._status = 0
        self.data = self.img_data
        self.all_data = self.img_all_data

    def img_data(self, index):
        if self._status:
            return (self._img[index, :],  self._label[index, :])
        else:
            ret_img = []
            ret_label = []
            for i in index:
                if self.train:
                    if not self._load[i]:
                        temp_img = imread(self.i_lines[i].strip().split()[0])
                        temp_img = imresize(temp_img, (224,224))
                        self._img[i] = temp_img
                        self._label[i] = [int(j) for j in self.l_lines[i].strip().split()[:]]
                        self._load[i] = 1
                        self._load_num += 1
                    ret_img.append(self._img[i])
                    ret_label.append(self._label[i])
                else:
                    self._label[i] = [int(j) for j in self.l_lines[i].strip().split()[:]]
                    temp_img = imread(self.i_lines[i].strip().split()[0])
                    if temp_img.shape[2]!=3:
                        temp_img = temp_img[0:3,:,:]
                    temp_img = imresize(temp_img, (224,224))
                    ret_img.append(temp_img)
                    ret_label.append([int(j) for j in self.l_lines[i].strip().split()[:]])
            if self._load_num == self.n_samples:
                self._status = 1
                self._img = np.asarray(self._img)
            return (ret_img, ret_label)

    def img_all_data(self):
        if self._status:
            return (self._img, self._label)
def import_train(config):
    return (dataset(config['img_tr'], config['lab_tr'], train=False))
def import_test(config):
    return (dataset(config['img_te'], config['lab_te'], train=False))
def import_db(config):
    return (dataset(config['img_db'], config['lab_db'], train=False))
