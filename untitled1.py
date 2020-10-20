# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:10:40 2019

@author: NRAD
"""

c=3065

import cv2

from glob import glob

from matplotlib import pyplot as plt

path=r'C:\Users\NRAD.NRAD-28\Desktop\tumordata\tumor\no_tumor\no\*.png'

pngs=glob(path)

for i in range(len(pngs)):
    
    img=plt.imread(pngs[i])
    
    name=r'C:\Users\NRAD.NRAD-28\Desktop\tumordata\tumor\no_tumor\id_' + str(c) + '.png'


    plt.imsave(name,img,cmap='Greys')
    
    c=c+1