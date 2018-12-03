import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.layers import Dense,Dropout
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"]='1'

base_model = VGG16(weights='imagenet',
                      include_top=False,
                      pooling=None,
                      input_shape=(32,32,3),classes=10)

for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
predictions = Dense(10,activation='softmax')(x)

model = Model(inputs=base_model.input,outputs=predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

(x_train, y_train),(x_test,y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.

# x_train_mean = np.mean(x_train,axis=0)
# x_train -= x_train_mean
# x_test_mean = np.mean(x_test,axis=0)
# x_test -= x_test_mean

datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.0
)
datagen.fit(x_train)
his = model.fit_generator(datagen.flow(x_train,y_train,
                                       batch_size=32),
                          steps_per_epoch=1000,
                          epochs=200,
                          validation_data=(x_test,y_test),
                          verbose=1,
                          workers=4)
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