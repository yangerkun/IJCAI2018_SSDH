import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

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
