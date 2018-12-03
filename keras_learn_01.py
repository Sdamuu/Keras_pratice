import os,shutil,random,glob
import cv2
import numpy as np
import pandas as pd
from tqdm import trange
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,GlobalAveragePooling2D,MaxPooling2D,ZeroPadding2D,BatchNormalization
import matplotlib.pyplot as plt
from keras.applications.mobilenet_v2 import MobileNetV2
import data_move
from keras import Model
resize = 224
# def load_data():
#     imgs = os.listdir("./train/")
#     num = len(imgs)
#     train_data = np.empty((5000, resize, resize, 3), dtype="int32")
#     train_label = np.empty((5000, ), dtype="int32")
#     test_data = np.empty((5000, resize, resize, 3), dtype="int32")
#     test_label = np.empty((5000, ), dtype="int32")
#     print("load training data")
#     for i in trange(5000):
#         if i % 2:
#             train_data[i] = cv2.resize(cv2.imread('./train/' + 'dog.' + str(i) + '.jpg'), (resize, resize))
#             train_label[i] = 1
#         else:
#             train_data[i] = cv2.resize(cv2.imread('./train/' + 'cat.' + str(i) + '.jpg'), (resize, resize))
#             train_label[i] = 0
#     print("\nload testing data")
#     for i in trange(5000, 10000):
#         if i % 2:
#             test_data[i-5000] = cv2.resize(cv2.imread('./train/' + 'dog.' + str(i) + '.jpg'), (resize, resize))
#             test_label[i-5000] = 1
#         else:
#             test_data[i-5000] = cv2.resize(cv2.imread('./train/' + 'cat.' + str(i) + '.jpg'), (resize, resize))
#             test_label[i-5000] = 0
#     return train_data, train_label, test_data, test_label
# train_data, train_label, test_data, test_label = load_data()
# train_data, test_data = train_data.astype('float32'), test_data.astype('float32')
# train_data,test_data = train_data/255.0, test_data/255.0
# # img = train_data[0]
# # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# # plt.imshow(img)
# # plt.axis("off")
# # plt.show()
# # 变为 one-hot 向量
# train_label = keras.utils.to_categorical(train_label,2)
# test_label = keras.utils.to_categorical(test_label,2)


# 搭建 AlexNet 网络结构

# 利用 flow_from_directory 进行训练数据的生成
# 首先进行数据增强操作
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#     './data/train/',
#     target_size=(224,224),
#     batch_size=64,
#     class_mode='binary')
# validation_generator = test_datagen.flow_from_directory(
#     './data/validation/',
#     batch_size=64,
#     target_size=(224,224),
#     class_mode='binary')
#
# model = Sequential()
# # level1
# model.add(Conv2D(filters=96,kernel_size=(11,11),
#                  strides=(4,4),padding='valid',
#                  input_shape=(resize,resize,3),
#                  activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(3,3),
#                        strides=(2,2),
#                        padding='valid'))
# # level_2
# model.add(Conv2D(filters=256,kernel_size=(5,5),
#                  strides=(1,1),padding='same',
#                  activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(3,3),
#                        strides=(2,2),
#                        padding='valid'))
# # layer_3
# model.add(Conv2D(filters=384,kernel_size=(3,3),
#                  strides=(1,1),padding='same',
#                  activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=384,kernel_size=(3,3),
#                  strides=(1,1),padding='same',
#                  activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=356,kernel_size=(3,3),
#                  activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(3,3),
#                        strides=(2,2),padding='valid'))
#
# # layer_4
# model.add(Flatten())
# model.add(Dense(4096,activation='relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(4096,activation='relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(1000,activation='relu'))
# model.add(Dropout(0.5))
#
# # output layer
# model.add(Dense(2))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])
# model.summary()
#
# # model.load_weights('./weights/model.h5')
#
# his = model.fit_generator(
#             data_move.train_flow,
#             steps_per_epoch=125,
#             epochs=50,
#             validation_generator=data_move.test_flow,
#             validation_steps=25)
# print(his.history.keys())
#
# plt.plot(his.history['acc'])
# plt.plot(his.history['val_acc'])
# plt.title('model_accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# plt.plot(his.history['loss'])
# plt.plot(his.history['val_loss'])
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# model.save('./weights/model.h5')
# 使用ResNet的结构，不包括最后一层
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling=None,
                   input_shape=(resize, resize, 3), classes = 2)
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights('./weights/catdogs_model.h5')
test_image = cv2.resize(cv2.imread('./43.jpg'),(224,224))
test_image = np.asarray(test_image.astype("float32"))
test_image = test_image/255.
test_image = test_image.reshape((1,224,224,3))
preds = model.predict(test_image)
if preds.argmax() == 0:
    print("cat")
else:
    print("dog")



































