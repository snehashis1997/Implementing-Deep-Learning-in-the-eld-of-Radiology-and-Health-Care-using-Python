# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 05:11:11 2019

@author: nrad
"""

from glob import glob
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray

def image_converter(path):
    pngs=glob(path)
    image_set=[]
    for i in range(len(pngs)):
        img=imread(pngs[i])
        gray=rgb2gray(img)
        #gray = cv2.resize(gray,(64,64))
        #gray=np.expand_dims(gray,2)
        image_set.append(gray)
    image_set=np.array(image_set)
    return(image_set)