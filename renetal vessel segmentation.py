
# coding: utf-8

# # unet implimentation in renetal vessel dataset

# In[8]:


from sklearn.model_selection import train_test_split
from glob import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
#from skimage.io import imread
import numpy as np
from keras.callbacks import History
import matplotlib.pyplot as plt


print('okay')


# In[ ]:


batchsize=32

print('okay')


# In[ ]:


# Set some parameters
IMG_WIDTH = 512/8
IMG_HEIGHT = 512/8
IMG_CHANNELS = 1

seed = 42

np.random.seed = seed

# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = (Model(inputs=[inputs], outputs=[outputs]))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()

print('okay')


# In[ ]:


path_image=r'C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\DeepVessel Results_MICCAI\DeepVessel Results_MICCAI\dataset\image\*.*'

path_label=r'C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\DeepVessel Results_MICCAI\DeepVessel Results_MICCAI\dataset\labels\*.*'


# In[ ]:


image_set=[]
mask_set=[]

pngs_image=glob(path_image_)
pngs_mask=glob(path_mask_)

for i in range(len(pngs_image)):
    img=imread(pngs_image[i])
    gray=rgb2gray(img)
    gray = resize(img,(64,64),anti_aliasing=True)
    gray=np.expand_dims(gray,2)
    image_set.append(gray)
    
    img1=imread(pngs_mask[i])
    gray1=rgb2gray(img1)
    gray1 = resize(img1,(64,64),anti_aliasing=True)
    gray1=np.expand_dims(gray1,2)
    mask_set.append(gray1)

image_set=np.array(image_set)
mask_set=np.array(mask_set)

print('okay')


# In[ ]:


mask_set[5].shape


# In[ ]:


image_set.shape


# In[ ]:


#image_set[0]


# In[ ]:


#mask_set.shape


# In[ ]:


model.fit(x=image_set, y=mask_set, batch_size=None, epochs=100, callbacks=None, validation_split=0.1, 
          validation_data=None,steps_per_epoch=200,validation_steps=3)


# In[ ]:


from keras.preprocessing import image

path_test=r'../input/membrane/membrane/test/0.png'
test_image=image.load_img(path_test,color_mode='grayscale')
test_image= image.img_to_array(test_image)

test_image=np.true_divide(test_image,[255,0],out=None)
test_image = resize(test_image,(64,64,1),anti_aliasing=True)


# In[ ]:


test_image.shape


# In[ ]:


results = model.predict(test_image,batch_size=None, verbose=0, steps=1)

results.shape


# In[ ]:


from matplotlib import pyplot as plt

results=np.reshape(results,(64,64))

results.shape


# In[2]:


plt.imshow(results,cmap='gray')
plt.show()

