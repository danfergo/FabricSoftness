from keras.preprocessing.image import ImageDataGenerator

# train
batch_size = 32
train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '/data/no_dots_gel/split/train/',
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

# validation
validation_generator = train_datagen.flow_from_directory(
        '/data/no_dots_gel/split/validation/',
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

from keras.applications.vgg16 import VGG16

vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224,3))

print (vgg16.input.shape)
print (vgg16.output.shape)



from keras.layers import Conv2D, Reshape, Dense, Activation, BatchNormalization

input = vgg16.input
# vgg16 goes here
fc1 = Conv2D(100, (7,7))(vgg16.output)
fc1_bn = BatchNormalization()(fc1)
fc1_act = Activation('relu')(fc1_bn)

fc2 = Conv2D(2, (1,1))(fc1_act)
fc2_bn = BatchNormalization()(fc2)
fc2_act = Activation('softmax')(fc2_bn)

output = Reshape((2,))(fc2_act)

print (fc1.shape)
print (fc2.shape)
print (output.shape)




from keras.models import Model
from keras.utils import print_summary

model = Model(inputs=input, outputs=output)

print_summary(model)


from keras.optimizers import Adadelta, SGD

optimizer = Adadelta(lr=0.50, rho=0.95, epsilon=None, decay=0.0)

model.compile(optimizer= optimizer,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


history = model.fit_generator(
        train_generator,
        steps_per_epoch=(3000/32),
        verbose=1,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=10)


import pickle

experiment_name = 'experiment_2'
base_path = '/data/models/'

model.save(base_path + '/model_' + experiment_name + '.h5')
model.save_weights(base_path + 'weights_' + experiment_name + '.h5')

pkl_file = open(base_path + 'history_' + experiment_name + '.pkl', 'wb')
pickle.dump(history.history, pkl_file)
pkl_file.close()


