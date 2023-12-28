# -*- coding:utf-8 -*-
# @FileName  :Corr-SPSS.py
# @Time      :2023/12/25/13:31
# @Author    :dengqi


import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
from scipy.linalg import logm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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


def predict_label(clf_name):
    clfs = {'Ridge': RidgeClassifier(), 'NB': GaussianNB(), "LR": LogisticRegression(),
            "LDA": LinearDiscriminantAnalysis(), 'SVC_linear': SVC(kernel='linear'),
            'SVC_Gaussian': SVC(kernel='rbf'), 'GMSVM': SVC(kernel='precomputed')}
    model = clfs[clf_name]
    if clf_name == "GMSVM":
        G_Tr = np.dot(train_cov, train_cov.T).T
        G_Te = np.dot(train_cov, test_cov.T)
        model.fit(G_Tr, train_label)
        predicted_label = model.predict(G_Te.T)
    else:
        model.fit(train_cov, train_label)
        predicted_label = model.predict(test_cov)
    return predicted_label


# %% LOAD DATA
sensor = [0, 1, 2, 12, 13, 3, 4, 5, 6, 7, 8]
K = len(sensor)
all_num = 100  # for each class
N = 512
overlap = int(N / 2)

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
load, speed = 0, 300
data_path = r"./dataset/负载{}/{}".format(load, speed)
data_dir = os.listdir(data_path)
print("load:", load, "speed:", speed)

all_data = []
for i in labels:
    data = loadmat(os.path.join(data_path, data_dir[i]))["Datas"][:all_num * overlap + N, sensor]
    print("loading file:", data_dir[i])
    all_data.append(data)
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
corr_names = ["Euclidean", "cross-corr", "cov", "pearson", "linear", "Gaussian", "Laplace"]
corr = corr_names[-1]  # change correlation algorithm
clf_names = ["Ridge", "NB", "LR", "LDA", "SVC_linear", "SVC_Gaussian", "GMSVM"]
clf = clf_names[-1]  # change classifer
kernelSize = 1.8  # Gaussian-0.1;Laplace-1.8
train_number = 5
repeat_num = 20

print("correlation algorithm:", corr)
print("classifer:", clf)
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
    predicted_label = predict_label(clf)
    Acc[i] = np.sum(test_label == predicted_label[test_id.astype(int)]) / len(test_label)
print('Acc:{}'.format(np.mean(Acc), 'Std:{}'.format(np.std(Acc))))
