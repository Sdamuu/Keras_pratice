from keras.preprocessing.image import ImageDataGenerator

train_dir = '/home/damu/PycharmProjects/Keras/data/train/'
test_dir = '/home/damu/PycharmProjects/Keras/data/validation/'

train_pic_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.5,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_pic_gen = ImageDataGenerator(rescale=1./255)
train_flow = train_pic_gen.flow_from_directory(train_dir,
                                               target_size=(224,224),
                                               batch_size=64,
                                               class_mode='categorical',
                                               shuffle=True)
test_flow = test_pic_gen.flow_from_directory(test_dir,
                                             target_size=(224,224),
                                             batch_size=64,
                                             class_mode='categorical')


print(train_flow.class_indices)
