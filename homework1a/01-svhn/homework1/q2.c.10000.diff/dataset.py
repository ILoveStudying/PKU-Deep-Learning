#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : dataset.py
# Purpose : generate samples for train and test
# Creation Date : 2019-02-19 10:35
# Last Modified :
# Created By : sunpeiqin
# =======================================

import os
import cv2
import numpy as np
from scipy import io as scio

from common import config


class Dataset():
    dataset_path = '../..'

    def __init__(self, dataset_name):
        self.minibatch_size = config.minibatch_size
        self.ds_name = dataset_name
        if config.use_extra_data:
            train_meta = ([os.path.join(self.dataset_path, 'train_32x32.mat'),
                           os.path.join(self.dataset_path, 'extra_32x32.mat')], 604388)
        else:
            train_meta = ([os.path.join(self.dataset_path, 'train_32x32.mat')], 73257)

        dataset_meta = {
            'train': train_meta,
            'test': ([os.path.join(self.dataset_path, 'test_32x32.mat')], 26032),
        }
        self.files, self.instances = dataset_meta[dataset_name]

    def load(self):
        datas_list, labels_list = [], []
        for f in self.files:
            samples = scio.loadmat(f)
            datas_list.append(samples['X'])
            labels_list.append(samples['y'])
        datas = np.concatenate(datas_list, axis=3)
        labels = np.concatenate(labels_list, axis=0)
        if self.ds_name == "train":
            datas, labels = self.shuffle_data(datas, labels)
            self.index = 0
            self.instances = 10000
            List = [1, 2, 3, 4, 5]
            self.count = [0 for i in range(len(List))]
            datas, labels = self.reduce_size(datas, labels, List, 6000)
        self.samples_mat = {
            'X': datas,
            'Y': labels,
        }
        return self

    def shuffle_data(self, data, label):
        index = np.arange(self.instances)
        np.random.shuffle(index)
        data = data[:, :, :, index]  # X_train是训练集，y_train是训练标签
        label = label[index]
        return data, label

    def reduce_size(self, data, labels, List, num):
        for j in range(data.shape[3]):
            if self.index < self.instances:
                if labels[j, :][0] == 10:
                    labels[j, :][0] = 0
                if labels[j, :][0] in List:
                    for i in range(len(List)):
                        if labels[j, :][0] == List[i] and self.count[i] < num:
                            self.count[i] += 1
                            data[:, :, :, self.index] = data[:, :, :, j]
                            labels[self.index, :] = labels[j, :][0]
                            self.index += 1
                            break
                else:
                    data[:, :, :, self.index] = data[:, :, :, j]
                    labels[self.index, :] = labels[j, :]
                    self.index += 1
            else:
                print(data[:, :, :, :self.instances].shape, labels[:self.instances, :].shape)
                return data[:, :, :, :self.instances], labels[:self.instances, :]

    @property
    def instances_per_epoch(self):
        return self.instances

    @property
    def minibatchs_per_epoch(self):
        return self.instances // config.minibatch_size

    def instance_generator(self):
        for i in range(self.instances):
            img = self.samples_mat['X'][:, :, :, i]
            label = self.samples_mat['Y'][i, :][0]
            if label == 10:
                label = 0
            img = cv2.resize(img, config.image_shape)
            yield img.astype(np.float32), np.array(label, dtype=np.int32)


if __name__ == "__main__":
    ds = Dataset('train')
    ds = ds.load()
    gen = ds.instance_generator()

    imggrid = []
    while True:
        for i in range(25):
            img, label = next(gen)
            cv2.putText(img, str(label), (0, config.image_shape[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            imggrid.append(img)

        imggrid = np.array(imggrid).reshape((5, 5, img.shape[0], img.shape[1], img.shape[2]))
        imggrid = imggrid.transpose((0, 2, 1, 3, 4)).reshape((5 * img.shape[0], 5 * img.shape[1], 3))
        cv2.imshow('', imggrid.astype('uint8'))
        c = chr(cv2.waitKey(0) & 0xff)
        if c == 'q':
            exit()
        imggrid = []
