# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 02:32:29 2019

@author: user
"""

import cv2
import h5py as h5
from glob import glob
import numpy as np
#import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

c=1

file_path=r"C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\brainTumorDataPublic_1766\mat_files/"

pngs=[]


for i in range(1,3065):
    
    pngs.append(file_path+str(i)+'.mat')

zero=np.zeros((len(pngs),2))

df = pd.DataFrame(zero, columns = ['PID', 'CLASS NO']) 


for i in range(len(pngs)):

    f=h5.File(pngs[i],'a')
    image=np.mat(f['/cjdata/image'])
    PID=np.array(f['/cjdata/PID'])
    label=np.array(f['/cjdata/label'])
    tumorBorder=np.mat(f['/cjdata/tumorBorder'])
    tumorMask=np.mat(f['/cjdata/tumorMask'])
    
    name = r"C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\brainTumorDataPublic_1766\image/id_" + str(c) +'.png'
    
    
    name_tumorMask = r"C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\brainTumorDataPublic_1766\mask/mask_id" + str(c) +'.png'
    
    
    df['PID'][i]=c

    df['CLASS NO'][i]=label
    
    c=c+1
    
   # plt.figure(figsize=(20,20))
    #plt.imshow(image,cmap='gray')
    #plt.tight_layout()
    #plt.axis('off')
    plt.imsave(name,image,cmap='Greys')
    
    #plt.figure(figsize=(20,20))
    #plt.imshow(tumorMask,cmap='gray')
    #plt.tight_layout()
    #plt.axis('off')
    plt.imsave(name_tumorMask,tumorMask,cmap='Greys')

    
csv_path_name=r"C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\brainTumorDataPublic_1766/mat2python_mri_tumor.csv"
df.to_csv(csv_path_name) 



path=r'C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\brain-mri-images-for-brain-tumor-detection\no\*.*'

pngs=glob(path)

for i in range(len(pngs)):
    
    img=cv2.imread(pngs[i])
    
    name=r'C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\brain-mri-images-for-brain-tumor-detection\newno\id_' + str(i) + '.png'
    
    plt.imsave(name,img,cmap='Greys')
    
data=pd.read_csv(r'C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\brain-mri-images-for-brain-tumor-detection\binary_tumor.csv')

zero=np.zeros((98,2))

df = pd.DataFrame(zero, columns = ['PID', 'CLASS NO']) 

data=data.append(df, ignore_index=False)


for i in range(98)
    
    
