# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 06:31:24 2019

@author: NRAD
"""

import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import skimage.io as io
import pandas as pd
import os
from glob import glob


dataset=[]

labels=[]


def index(a_list, value):
    try:
        return a_list.index(value)
    
    except ValueError:
        return None

def image_mha_information_read(path_image,path_mask):
        
    image= io.imread(path_image, plugin='simpleitk')
    
    mask=io.imread(path_mask,plugin='simpleitk')
    
    #print(image.shape)
    store=[]
    
    for i in range(image.shape[0]):
        
        img=image[:][i]
        
        values = list(set(img.flatten()))
        
        #store.append(i)
        
        length=len(values)
        
        if(length > 1):
            
            dataset.append(img)
            
            store.append(i)
            
    
    for i in range(len(store)):
       
       msk=mask[:][store[i]]
       
       labels.append(msk)
       
            
root_path=r'C:\Users\NRAD.NRAD-28\Downloads\BRATS2013\Image_Data'

mask_files = []

image_files=[]

# r=root, d=directories, f = files


for r, d, f in os.walk(root_path):
    for file in f:
        if '.mha' and 'OT' in r:
            mask_files.append(os.path.join(r, file))
            

for r, d, f in os.walk(root_path):
    for file in f:
        if '.mha' and 'T1c' in r:
            
            image_files.append(os.path.join(r, file))    
            
            

for i in range(len(image_files)):
    
    mask_path=mask_files[i]
    
    img_path=image_files[i]
    
    if (index(img_path,'N4ITK') == None and index(mask_path,'N4ITK') == None):
        
        image_mha_information_read(img_path,mask_path)
        

        
np.save(r'D:\brats_segmentation\npy_file_t1c\dataset',dataset)

np.save(r'D:\brats_segmentation\npy_file_t1c\mask',labels)

d=np.load(r'D:\brats_segmentation\npy_file_t1c\dataset.npy')

lbl=np.load(r'D:\brats_segmentation\npy_file_t1c\mask.npy')


            
        
