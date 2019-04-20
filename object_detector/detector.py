#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:33:30 2019

@author: maqianli
"""
import numpy as np 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
from config import *
from skimage import color
import matplotlib.pyplot as plt 
import os 
import glob

model_path = '../data/models/svm_model'
def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield(x, y, image[y : y + window_size[1], x:x + window_size[0]])
            
def detector(filename):
    im = cv2.imread(filename)
    im = imutils.resize(im, width = min(400, im.shape[1]))
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    downscale = 1.25
    
    clf = joblib.load(model_path)
    detections = []
    scale = 0

    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            im_window = color.rgb2gray(im_window)
            fd = hog(im_window, orientations=9, pixels_per_cell=[8,8], cells_per_block=[2,2], 
                 visualise=False, transform_sqrt=True)
            
            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)

           
            if pred == 1:
                if clf.decision_function(fd) > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale ** scale)),
                                       clf.decision_function(fd),
                                       int(min_wdw_sz[0] * (downscale ** scale)),
                                       int(min_wdw_sz[1] * (downscale ** scale))))
                    
        scale += 1
    
    clone = im.copy()
    
    for(x_t1, y_t1, _, w, h) in detections:

        cv2.rectangle(im, (x_t1, y_t1), (x_t1 + w, y_t1 + h), (0, 255, 0), thickness = 2)
    
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print("sc: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
    #print("shape, ",pick.shape)
    
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    plt.axis("off")
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title('Raw Detection before NMS')
    plt.show()
    
    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()

def test_folder(foldername):
    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        print(filename)
        detector(filename)
        
if __name__ == '__main__':
    foldername = 'test_image'
    test_folder(foldername)
    
    
    
        
    
    
    
    
    
    
    
    
        
        
        
        
        
