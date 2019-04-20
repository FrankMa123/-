#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:04:30 2019

@author: maqianli
"""
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import glob
import os
from config import *

def extract_features():
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)
        
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        im = imread(im_path, as_grey=True)
        print(im.shape)
        fd = hog(im, orientations=9, pixels_per_cell=[8,8], cells_per_block=[2,2], 
                 visualise=False, transform_sqrt=True)
        print(fd.shape)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
       
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        im = imread(im_path, as_grey=True)
        fd = hog(im, orientations=9, pixels_per_cell=[8,8], cells_per_block=[2,2], 
                 visualise=False, transform_sqrt=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(neg_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    
    print('completed')


if __name__=='__main__':
    extract_features()

