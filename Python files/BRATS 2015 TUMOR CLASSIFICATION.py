import numpy as np
seed = 42
np.random.seed(seed)

from sklearn.metrics import confusion_matrix
import cv2
import copy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,BatchNormalization,GaussianNoise
from keras.layers import MaxPool2D,ZeroPadding2D,MaxPooling2D
from keras.layers import Flatten,Activation
from keras.layers import Dense,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping
from keras import initializers
import numpy as np
from keras import regularizers


from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import auc,roc_curve

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from glob import glob
import os

target=512

data_path=r"../input/tumor/tumor/image/image/id_"
pngs=[]
#pngs=glob(data_path)

for i in range(1,3065):
    pngs.append(data_path+str(i)+'.png')

len(pngs)

pngs[0]

img=cv2.imread(r"../input/tumor/tumor/image/image/id_1.png")
len(img)

import pandas as pd

df=pd.read_csv("../input/tumor/tumor/mat2python_mri_tumor.csv")
from sklearn.model_selection import train_test_split

y_true=[]

for i in range(3064):
    y_true.append(int(df['CLASS NO'][i]))

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

#SVG(model_to_dot(model).create(prog='dot', format='svg'))

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

encoder = LabelEncoder()
encoder.fit(y_true)
y_true = encoder.transform(y_true)
y_true = to_categorical(y_true)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./ 255)

batchsize =16

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)

# we put our call backs into a callback list
callbacks = [earlystop, reduce_lr]

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Input,GlobalAveragePooling2D
from keras.models import Model

target=224

base_model = InceptionV3(weights='imagenet',input_shape = (target, target, 3), include_top=False)

for (i,layer) in enumerate(base_model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
x = base_model.output
x = Flatten(name = "flatten")(x)# let's add a fully-connected layer
x = Dense(512,activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(256,kernel_regularizer=regularizers.l2(0.0001),activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

for (i,layer) in enumerate(base_model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)

dataset=list()

for i in range(len(pngs)):
    
    img=cv2.imread(pngs[i])
    #img=cv2.equalizeHist(img)
    img=cv2.resize(img,(target,target))
    dataset.append(img)

dataset=np.array(dataset)

len(dataset)

X_trainset,X_test1 ,y_trainset,y_test1= train_test_split(dataset,y_true,test_size=0.4, shuffle=True,random_state=seed)

X_trainset = X_trainset.reshape(-1,target,target,3)
X_test1 = X_test1.reshape(-1,target,target,3)

X_train1.shape,len(y_train1)

X_train1,X_valid1,y_train1,y_valid1 = train_test_split(X_trainset,y_trainset,test_size=0.3, shuffle=False,random_state=seed)

X_valid1=X_valid1/np.max(X_valid1)

X_train1.shape[0]

model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['acc'])

output1 =model.fit_generator(train_datagen.flow(X_train1, y_train1, batch_size=batchsize),steps_per_epoch=X_train1.shape[0]//batchsize, 
                                 epochs=10, verbose=1,
                                 validation_data=test_datagen.flow(X_valid1,y_valid1), validation_steps=X_valid1.shape[0],shuffle=True)


plt.plot(output1.history['acc'])
plt.plot(output1.history['val_acc'])
plt.title('classifier accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(output1.history['loss'])
plt.plot(output1.history['val_loss'])
plt.title('classifier loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

for (i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)

N=104

#for layer in model.layers[:N]:
   #layer.trainable = False
for layer in model.layers[:]:
   layer.trainable = True


for (i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)

model.compile(optimizer=Adam(0.00001), loss='categorical_crossentropy', metrics=['acc'])

output2 =model.fit_generator(train_datagen.flow(X_train1, y_train1, batch_size=batchsize),steps_per_epoch=X_train1.shape[0]//batchsize, 
                                 epochs=20, verbose=1,
                                 validation_data=(X_valid1,y_valid1), validation_steps=X_valid1.shape[0],shuffle=True)


# summarize history for accuracy
plt.plot(output2.history['acc'])
plt.plot(output2.history['val_acc'])
plt.title('classifier accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(output2.history['loss'])
plt.plot(output2.history['val_loss'])
plt.title('classifier loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


score=model.evaluate_generator(test_datagen.flow(X_test1, y_test1, batch_size=batchsize),steps=1)

#predict_classes=model.predict_classes(X_test)

score[0],score[1]

y_hat=model.predict(X_test1)

y_hat

cm=confusion_matrix(y_true,y_hat)

report=classification_report(y_test1,y_hat)

print(cm)
print('\n')

print(classi_report)
print('\n')
