#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:56:42 2019

@author: maqianli
"""
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import glob
import os
from config import *
import numpy as np



def train_svm():
    pos_feat_path = '../data/features/pos'
    neg_feat_path = '../data/features/neg'
    model_path = '../data/models/svm_model'
    fds, labels = [], []
    
    for feat_path in glob.glob(os.path.join(pos_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)
    
    for feat_path in glob.glob(os.path.join(neg_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    
    clf = LinearSVC()
    clf.fit(fds, labels)
    
    if not os.path.isdir(os.path.split(model_path)[0]):
        os.makedirs(os.path.split(model_path)[0])
    #print('Classifier save to {}'.format(model_path))
    joblib.dump(clf, model_path)
    print('Classifier save to {}'.format(model_path))
    
if __name__=='__main__':
    train_svm()
