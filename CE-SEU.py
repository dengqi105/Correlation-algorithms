# -*- coding:utf-8 -*-
# @FileName  :CE-SEU.py
# @Time      :2023/12/15/13:02
# @Author    :dengqi


import numpy as np
import pandas as pd
import os

from scipy.linalg import logm
from sklearn import svm


def get_sample(labels, train_number, class_num):
    Train_label = []
    Train_index = []
    for ii in range(class_num):
        labels_all = np.reshape(labels, -1, order='F')
        index_ii = np.where(labels_all == ii)[0]
        class_ii = np.ones(len(index_ii), dtype=int) * ii
        Train_label = np.concatenate((Train_label, class_ii))
        Train_index = np.concatenate((Train_index, index_ii))
    Train_index = Train_index.astype(int)
    trainall = np.zeros((2, len(Train_index)))
    trainall[0, :] = Train_index
    trainall[1, :] = Train_label
    indexes = []
    for i in range(class_num):
        Class_Index = np.where(Train_label == i)[0]
        Random_num = np.random.permutation(len(Class_Index))
        Random_num = np.squeeze(Random_num.T)
        Random_Index = Class_Index[Random_num]
        Train_index = Random_Index[:train_number]
        indexes.extend(Train_index)
    indexes = np.array(indexes)
    train_LI = trainall[:, indexes]
    test_LI = trainall.copy()
    test_LI = np.delete(test_LI, indexes, axis=1)
    return train_LI, test_LI


def get_corr(X, Y, corr, kernel_size):
    if corr == "Laplace":
        corr_value = np.exp(-np.sqrt(np.sum((X - Y) ** 2)) / kernel_size)
    if corr == "Gaussian":
        corr_value = np.sum(np.exp(-((X - Y) ** 2) / (2 * kernel_size ** 2)))
    if corr == "cross-corr":
        corr_value = np.correlate(X, Y, mode="valid")
    if corr == "cov":
        corr_value = np.cov(X, Y)[0, 0]
    if corr == "pearson":
        corr_value = np.corrcoef(np.stack((X, Y), axis=0))[0, 1]
    if corr == "Euclidean":
        corr_value = np.sqrt(np.sum((X - Y) ** 2))
    if corr == "linear":
        corr_value = np.dot(X, Y.T)
    return corr_value


def corr_matrix(data, corr, kernel_size):
    dim = data.shape[1]
    corr_M = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i + 1):
            corr_M[i, j] = get_corr(data[:, i], data[:, j], corr, kernel_size)
    corr_M = (corr_M + corr_M.T) - np.eye(np.shape(corr_M)[0]) * np.tile(np.diag(corr_M), (np.shape(corr_M)[1], 1))
    return corr_M


def get_corr_matrix(sensor_seg, K, corr, kernel_size):
    tol = 1e-3
    shape = np.shape(sensor_seg)
    all_samples = np.zeros((K, K, shape[0], shape[3]), dtype=np.float32)
    for i in range(shape[3]):
        for j in range(shape[0]):
            K_segments = sensor_seg[j, :, :, i]
            corr_M = corr_matrix(K_segments, corr, kernel_size)
            eye = np.eye(np.shape(corr_M)[0])
            all_samples[:, :, j, i] = logm(corr_M + tol * eye * np.trace(corr_M))
    return all_samples


# %% LOAD DATA
sensor = [0, 1, 2, 3, 4, 5, 6, 7]
K = len(sensor)
all_num = 500  # for each class
N = 2048
overlap = int(N / 2)

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
dataset_name = ["gearset", "bearingset"][0]  # change dataset
print("dataset name:", dataset_name)
data_path = r"./data/{}/".format(dataset_name)
data_dir = os.listdir(data_path)

all_data = []
for i in labels:
    datas = []
    sep = "\t" if data_dir[i] != "ball_20_0.csv" else ","
    data = pd.read_csv(data_path + data_dir[i], header=None, index_col=None, low_memory=False, sep=sep)
    print("loading file:", data_dir[i])
    for j in range(K):
        d = pd.to_numeric(data.iloc[16:all_num * overlap + N, j]).values
        datas.append(d)
    datas = np.row_stack(datas).T
    all_data.append(datas)
all_data = np.stack(all_data, axis=-1)

segs = []
for j in range(all_num):
    sample = all_data[j * overlap:j * overlap + N, :, :]
    segs.append(sample)
segs = np.stack(segs, axis=0)

datalabel = np.zeros((all_num, len(labels)), dtype=np.int32)
for i in range(len(labels)):
    for j in range(all_num):
        datalabel[j, i] = labels[i]

# %% TRAIN AND TEST
corr_set = ["Euclidean", "cross-corr", "cov", "pearson", "linear", "Gaussian", "Laplace"]
corr = corr_set[1]  # change correlation algorithm
kernelSize = 1.8  # Gaussian-0.1;Laplace-1.8
train_number = 5
repeat_num = 20

print("correlation algorithm:", corr)
print("train_number:", train_number)

all_M = get_corr_matrix(segs, K, corr=corr, kernel_size=kernelSize)
size = np.shape(all_M)
corr_feature = np.reshape(all_M, (size[0], size[1], size[2] * size[3]), order='F')
datalabel = datalabel[0:size[2], :]

Acc = np.zeros(repeat_num)
for i in range(repeat_num):
    train_SL, test_SL = get_sample(datalabel, train_number, len(labels))
    train_id = train_SL[0, :].astype(int)
    train_label = train_SL[1, :]
    test_id = test_SL[0, :]
    test_label = test_SL[1, :].astype(int)
    train = corr_feature[:, :, train_id]
    train_cov = np.transpose(np.reshape(train, (K * K, train.shape[2])))
    test = corr_feature
    test_cov = np.transpose(np.reshape(test, (K * K, test.shape[2])))
    G_Tr = np.dot(train_cov, train_cov.T).T
    G_Te = np.dot(train_cov, test_cov.T)
    model = svm.SVC(kernel='precomputed')
    model.fit(G_Tr, train_label)
    predict_label = model.predict(G_Te.T)
    Acc[i] = np.sum(test_label == predict_label[test_id.astype(int)]) / len(test_label)
print(f'Acc: {np.mean(Acc)}')
