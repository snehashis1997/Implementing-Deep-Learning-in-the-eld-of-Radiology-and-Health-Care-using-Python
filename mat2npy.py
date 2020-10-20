# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 01:26:08 2019

@author: NRAD
"""

import random
import h5py as h5
from glob import glob
import numpy as np
from matplotlib import pyplot as plt

file_path=r"D:\tumor_from_nat_files\mat_files\*.mat"

pngs=glob(file_path)

dataset=[]

mask_set=[]

for i in range(len(pngs)):
    
    f=h5.File(pngs[i],'a')
    
    image=np.mat(f['/cjdata/image'])
    
    tumorMask=np.mat(f['/cjdata/tumorMask'])
    
    dataset.append(image)
    
    mask_set.append(tumorMask)
    
dataset=np.array(dataset)

mask_set=np.array(mask_set)


np.save(r'D:\tumor_from_nat_files\npy_file\dataset.npy',dataset)

np.save(r'D:\tumor_from_nat_files\npy_file\mask_set.npy',mask_set)  


dataset=np.load(r'D:\tumor_from_nat_files\npy_file\dataset.npy',allow_pickle=True) 

mask_set=np.load(r'D:\tumor_from_nat_files\npy_file\mask_set.npy',allow_pickle=True)  


c=random.randint(0,100)

plt.imshow(dataset[c],cmap='gray')

plt.show()

plt.imshow(mask_set[c],cmap='gray')

plt.show()


 
