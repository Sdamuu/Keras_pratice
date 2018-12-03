import os,shutil,random,glob
import cv2
import numpy as np
import pandas as pd
from tqdm import trange
import keras

from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3

from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Dropout
from keras.models import Model,load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

resize = 224
def load_data():
    imgs = os.listdir("./train/")
    num = len(imgs)
    print(num)
    train_data = np.empty((20000,resize,resize,3),dtype="int32")
    train_label = np.empty((20000,),dtype="int32")
    # test_data = np.empty((12500,resize,resize,3),dtype="int32")
    # test_label = np.empty((12500,),dtype="int32")
    for i in trange(10000):
        train_data[i] = cv2.resize(cv2.imread('./train/'+'dog.'+str(i)+'.jpg'),
                                       (resize,resize))
        train_label[i] = 1
    for i in trange(10000,20000):
        train_data[i] = cv2.resize(cv2.imread('./train/'+'cat.'+str(i-10000)+'.jpg'),
                                       (resize,resize))
        train_label[i] = 0
    # for i in trange(5000,10000):
    #     if i%2:
    #         test_data[i-5000] = cv2.resize(cv2.imread('./train/'+'dog.'+str(i)+'.jpg'),
    #                                   (resize,resize))
    #         test_label[i-5000] = 1
    #     else:
    #         test_data[i-5000] = cv2.resize(cv2.imread('./train/'+'cat.'+str(i)+'.jpg'),
    #                                   (resize,resize))
    #         test_label[i-5000] = 0
    # return train_data,train_label,test_data,test_label
    return train_data, train_label

train_data, train_label = load_data()
train_data= train_data.astype('float32')
train_data = train_data/255.0
train_label = keras.utils.to_categorical(train_label, 2)
# test_label = keras.utils.to_categorical(test_label,2)

base_model = VGG16(weights='imagenet', include_top=False, pooling=None,
                   input_shape=(resize, resize, 3), classes = 2)
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
his = model.fit(train_data,train_label,
          batch_size=64,
          epochs=50,
          validation_split=0.2,
          shuffle=True)

print(his.history.keys())

plt.plot(his.history['acc'])
plt.plot(his.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('./weights/model_resnet50.h5')


























