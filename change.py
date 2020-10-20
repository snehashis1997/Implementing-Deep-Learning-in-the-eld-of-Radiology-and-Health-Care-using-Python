# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 08:41:11 2019

@author: NRAD
"""

#import cv2

from glob import glob


    
import pandas as pd

import numpy as np

c=3065

df=pd.read_csv(r'C:\Users\NRAD.NRAD-28\Desktop\tumordata\tumor\mat2python_mri_tumor.csv')

pngs_no=glob(r'C:\Users\NRAD.NRAD-28\Desktop\tumordata\tumor\no_tumor\no\*.png')

no=np.zeros((len(pngs_no),2))

df=df.drop(columns=['Unnamed: 0'])

no=pd.DataFrame(no,columns=['PID','CLASS NO'])

for i in range(110):
    
    no['PID'][i]=c
    
    c=c+1

df_new=pd.concat([df,no],ignore_index=True)

#df_new=df_new.drop(columns=[0,1])

df_new.to_csv(r'C:\Users\NRAD.NRAD-28\Desktop\tumordata\tumor\with_12_no_tumor.csv')