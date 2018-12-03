import keras
import os
from keras.callbacks import TensorBoard
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D,BatchNormalization,GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from keras.models import Model,load_model
import data_move
os.environ["CUDA_VISIBLE_DEVICES"]='1'
resize = 224
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/1',
                                         histogram_freq= 0,
                                         write_graph=True,
                                         write_images=True)


base_model = ResNet50(weights='imagenet', include_top=False, pooling=None,
                   input_shape=(resize, resize, 3), classes = 2)
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

his = model.fit_generator(data_move.train_flow,
                    steps_per_epoch=100,
                    epochs=50,
                    verbose=1,
                    validation_data=data_move.test_flow,
                    validation_steps=500,
                    callbacks=[tbCallBack])

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


model.save('./weights/catdogs_model.h5')
