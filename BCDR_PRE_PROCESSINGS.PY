# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 02:39:05 2019

@author: NRAD
"""

import cv2
import os

from matplotlib import pyplot as plt

path=r'C:\Users\NRAD.NRAD-28\Desktop\BCDR-F03_dataset-20190626T062636Z-001'

files_list=[]

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".tif"):
             files_list.append(os.path.join(root, file))
             
             #print(os.path.join(root, file))
             
             
for i in range(len(files_list)):
    
    img=cv2.imread(files_list[i])


    shape=img.shape[0]

    img=img[200:shape,:,:]
    
    cv2.imwrite(files_list[i],img)
    

import pandas as pd

df=pd.read_csv(r"C:\Users\NRAD.NRAD-28\Desktop\BCDR-F03_dataset-20190626T062636Z-001\BCDR-F03_dataset\BCDR-F03\bcdr_f03_features.csv")

truth_values=df['classification']

y_truth_list=[]

for i in range(668):
    
    index1=files_list[i].rindex('_')
    
    index2=files_list[i].rindex('_',0,index1)
    
    index3=files_list[i].rindex('_',0,index2)
    
    view=files_list[i][index1+1:-4]
    
    study_id=files_list[i][index2+1:index2+5]
    
    patient_id=files_list[i][index3+1:index3+4]
    
    #print(view)
    
    if(df['patient_id'][i] == patient_id and df['study_id'][i]==study_id):
        
        y_truth_list.append(df['classification'][i])