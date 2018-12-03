import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import SGD
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="1"

"""基于多层感知机 (MLP) 的 softmax 分类"""
# # 生成虚拟数据
# import numpy as np
# x_train = np.random.random((1000, 20))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
# x_test = np.random.random((100, 20))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
# model = Sequential()
# # Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
# # 在第一层必须指定所期望的输入数据尺寸：
# # 在这里，是一个 20 维的向量。
# model.add(Dense(64,activation='relu',input_dim=20))
# model.add(Dropout(0.5))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10,activation='softmax'))
#
# sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,
#           nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])
# his = model.fit(x_train,y_train,
#           epochs=20,
#           batch_size=128)
# print(his.history.keys())
# plt.plot(his.history['acc'])
# plt.title('model_accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# plt.plot(his.history['loss'])
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# score = model.evaluate(x_test,y_test,batch_size=128)
# print(score)

"""类似 VGG 的卷积神经网络"""

# 生成虚拟数据
x_train = np.random.random((100,100,100,3))
y_train = keras.utils.to_categorical(np.random.randint(10,size=(100,1)),num_classes=10)
x_test = np.random.random((20,100,100,3))
y_test = keras.utils.to_categorical(np.random.randint(10,size=(20,1)),num_classes=10)


# 输入 3 通道 100×100 像素图像 -> (100,100,3) 张量
# 使用 32 个大小为 3×3 的卷积滤波器
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
model.add(Conv2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add((Dense(10,activation='softmax')))

sgd = SGD(lr=0.01,momentum=0.9,decay=1e-6,
          nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer='sgd',
              metrics=['accuracy'])

his = model.fit(x_train,y_train,batch_size=32,epochs=10)
print(his.history.keys())
plt.plot(his.history['acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(his.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

score = model.evaluate(x_test, y_test, batch_size=32)























































