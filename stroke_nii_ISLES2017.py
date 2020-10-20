# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 03:13:24 2019

@author: NRAD
"""

import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import nibabel as nib

path_image=r'C:\Users\NRAD.NRAD-28\Downloads\group3\group3\01\01_preop_mri.mnc'

path_test=r'C:\Users\NRAD.NRAD-28\Desktop\nii\la_003.nii'

#image= io.imread(path_image, plugin='simpleitk')

data=nib.load(path_image)

dat=data.get_data()

dataset=dat[1,:,:]

plt.imshow(dataset,cmap='gray')
plt.show()

image= io.imread(path_test, plugin='simpleitk')
dataset=image[100,:,:]

plt.imshow(dataset,cmap='gray')
plt.show()