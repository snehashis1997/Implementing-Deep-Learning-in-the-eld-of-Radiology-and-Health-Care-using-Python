# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 03:23:46 2019

@author: NRAD
"""

import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import shutil

def index(a_list, value):
    try:
        return a_list.index(value)
    
    except ValueError:
        return None
    

def image_saver(path,count,img_path):
    
    img=cv2.imread(path,0)

    values = list(set(img.flatten()))
    
    length=len(values)
    
    path_img=img_path[:img_path.rindex('\\')+1] + str(path[path.rindex('\\')+1:-4]) + '.png'
    
    destination_folder_mask=r"D:\brats_segmentation\T1C\mask\\" + str(count) + '.png'
    
    destination_folder_img=r"D:\brats_segmentation\T1C\images\\" + str(count) + '.png'


    if(length > 1):
        
        shutil.copyfile(path,destination_folder_mask)
        
        shutil.copyfile(path_img,destination_folder_img)

        
        
path = r'D:\BRATS2013\Image_Data'
        

files = []

T2_files=[]

# r=root, d=directories, f = files


for r, d, f in os.walk(path):
    for file in f:
        if 'MR_T1c' in r:
            T2_files.append(os.path.join(r, file))
            

for r, d, f in os.walk(path):
    for file in f:
        if 'XX.XX.OT' in r:
            
            files.append(os.path.join(r, file))
            
c=0

count=0

for i in range(len(files)):
    
    path=files[i]
    
    img_path=T2_files[c]
    
    if (index(img_path,'N4ITK') == None and index(path,'N4ITK') == None):
        
        image_saver(path,count,img_path)
        
        count=count+1
        
    
    c=c+1