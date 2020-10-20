# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 03:29:47 2019

@author: NRAD
"""

import numpy as np

import pandas as pd

import os

path=r'D:\medical datasets\breast-cancer-dataset-from-breakhis\fold1\test\40X\B_40X'
      
pngs_test = []

# r=root, d=directories, f = files


for r, d, f in os.walk(path):
    for file in f:
        pngs_test.append(file)


print(len(pngs_test))

zeros=np.zeros((len(pngs_test),2))

df=pd.DataFrame(zeros,columns=['binary','multiclass'])

print('okay')

for i in range(len(pngs_test)):
    
    index1=pngs_test[i].index('_')

    index2=pngs_test[i].index('_',index1+1)

    index3=pngs_test[i].index('-')
    
    binary_class=pngs_test[i][index1+1]
    
    multiclass=pngs_test[i][index1+3:index3]
    
    if binary_class=='B':
    
        df['binary'][i]=0
    
    else:
    
        df['binary'][i]=1
        
    if multiclass==str('A'):
    
        df['multiclass'][i]=2
    
    elif multiclass==str('F'):
    
        df['multiclass'][i]=3
    
    elif multiclass==str('PT'):
    
        df['multiclass'][i]=4
    
    elif multiclass==str('TA'):
    
        df['multiclass'][i]=5
    
    elif multiclass==str('DC'):
    
        df['multiclass'][i]=6
    
    elif multiclass==str('LC'):
    
        df['multiclass'][i]=7
    
    elif multiclass==str('MC'):
    
        df['multiclass'][i]=8
    
    elif multiclass==str('PC'):
    
        df['multiclass'][i]=9


name=r'C:\Users\NRAD.NRAD-28\Desktop\codec_test_csv.csv'

df.to_csv(name)