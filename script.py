# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:21:39 2019

@author: nrad
"""

from sklearn.metrics import confusion_matrix
import cv2
import copy
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,BatchNormalization,GaussianNoise
from tensorflow.keras.layers import MaxPooling2D,ZeroPadding2D
from tensorflow.keras.layers import Flatten,Activation
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
import numpy as np
from tensorflow.keras import regularizers

from glob import glob
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import auc,roc_curve


batchsize = 32
seed = 0

path_valid=r'C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\brain-mri-images-for-brain-tumor-detection\valid'
path_test=r'C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\brain-mri-images-for-brain-tumor-detection\test'
path_train=r'C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\brain-mri-images-for-brain-tumor-detection\train'

train_datagen = ImageDataGenerator(
        rescale=1./255,rotation_range=90,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

height=64
width=64

train_generator = train_datagen.flow_from_directory(
        path_train,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(path_valid,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(path_test,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


model=Sequential()
model.add(GaussianNoise(0.05))
model.add(Convolution2D(8,kernel_size=(3,3),
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.0001),input_shape=(64,64,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Convolution2D(8,kernel_size=(3,3),
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(rate=0.5))

model.add(Flatten())
model.add(Dense(32,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)))

model.add(BatchNormalization())
model.add(Dropout(rate=0.8))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

output= model.fit_generator(train_generator, steps_per_epoch=182/batchsize, epochs=50,
                      callbacks=None, validation_data=validation_generator, 
                      validation_steps=46)


plt.plot(output.history['acc'])
plt.plot(output.history['val_acc'])
plt.title('classifier  based accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(output.history['loss'])
plt.plot(output.history['val_loss'])
plt.title('classifier based loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



zero=np.zeros((10,1))

ones=np.ones((15,1))

y_true=np.concatenate((zero,ones))


y_hat=[]
path_test_png=r'C:\Users\NRAD\Desktop\snehashis internship 2019\medical dataset\brain-mri-images-for-brain-tumor-detection\test\*.*'
from tensorflow.keras.preprocessing import image
pngss=glob(path_test_png)
for i in range(len(pngss)):
    test_image = image.load_img(pngss[i] ,target_size= (64,64))
    arr = np.array(test_image)
    arr = np.true_divide(arr,[255.0],out=None)


# Changing the input of the size...
    test_image = image.img_to_array(arr)

# Adding a new dimension (the placement of the image in the batchsize)
    test_image = np.expand_dims(test_image, axis=0)

    predic_classes = model.predict_classes(test_image)
    y_hat.append(predic_classes[0])

cm=confusion_matrix(y_true,y_hat)
print(cm)

y_true=list(y_true)
y_hat=list(y_hat)

#classi_report=classification_report(y_true, y_hat)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_hat)
auc_curve=auc(fpr_keras, tpr_keras)

print('auc score is:'+str(auc_curve))

test_accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[1,0]+cm[0,1])

print('predict accuracy:'+str(test_accuracy))

model.save_weights(r'C:\Users\NRAD\Desktop\snehashis internship 2019\model\model_brain_yes_or_n0_auc:0.84.h5')