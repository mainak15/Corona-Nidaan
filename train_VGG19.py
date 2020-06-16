"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.

Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""
from keras.applications.vgg19 import VGG19
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization,Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
import time
import os.path
import itertools
import cv2
from glob import glob
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
import math 
import keras
from tensorflow.keras.regularizers import l2

#data = DataSet()
train_data_dir = 'data/train3'
valid_data_dir = 'data/test3'
# Helper: Save the model.
batch_size=32
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'VGG19_1.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
#early_stopper = EarlyStopping(patience=10)
early_stopper = EarlyStopping(monitor='val_loss', patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('data', 'logs', 'VGG19_1' + '-' + 'training-' + \
        str(timestamp) + '.log'))

def get_generators():
    '''
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)'''

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        classes=['COVID-19','normal','pneumonia'],
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        valid_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        classes=['COVID-19','normal','pneumonia'],
        class_mode='categorical')

    return train_generator, validation_generator

def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = VGG19(weights=weights, include_top=False)
    #for layer in base_model.layers:
        #print(layer.name)
        #layer.trainable = False
    print("Number of layers in the base model: ", len(base_model.layers))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dropout(0.4)(x)
    x = Dense(1024, activation='relu',kernel_initializer='he_uniform',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(x)
    # and a logistic layer
    predictions = Dense(3, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    print(model.summary())
    return model

def vgg16():
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=(224,224,3), activation='relu', name='conv_1'))
    model.add(Conv2D(32, (3,3), activation='relu', name='conv_2'))
    model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_1'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding='same', activation='relu', name='conv_3'))
    model.add(Conv2D(64, (3,3), activation='relu', name='conv_4'))
    model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_2'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3,3), padding='same', activation='relu', name='conv_5'))
    model.add(Conv2D(128, (3,3), activation='relu', name='conv_6'))
    model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_3'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='dense_2'))
    model.add(Dense(3, activation='softmax', name='predictions'))
    
    # compile the model (should be done *after* setting layers to non-trainable)
    optimizer = Adam(lr=1e-4, decay=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model
epochs = 25
initial_lrate = 0.01
def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)

lr_sc = LearningRateScheduler(decay, verbose=1)
'''
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)'''
def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:15]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    
    
    #for layer in model.layers:
    #    print(layer, layer.trainable)
    

    return model

#def train_model(model, nb_epoch, generators, lrate,callbacks=[]):
def train_model(model, nb_epoch, generators, callbacks=[]):
    #train_generator, validation_generator = generators
    from sklearn.utils import class_weight
    import numpy as np
    from matplotlib import pyplot

    train_generator, validation_generator = generators

    #optimizer = Adam(lr=lrate)
    optimizer=SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes)
    #class_weights = dict(enumerate(class_weights))
    print(class_weights)
    his=model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames) // batch_size,
        validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames) // batch_size,
        epochs=nb_epoch,
        class_weight=class_weights,
        callbacks=callbacks)
    
    fig, axs = plt.subplots(1, 2, figsize = (15, 4))
    training_loss = his.history['loss']
    validation_loss = his.history['val_loss']
    training_accuracy = his.history['accuracy']
    validation_accuracy = his.history['val_accuracy']
    epoch_count = range(1, len(training_loss) + 1)
    #N=num_epochs
    axs[0].plot(epoch_count, training_loss, 'r--')
    axs[0].plot(epoch_count, validation_loss, 'b-')
    axs[0].legend(['Training Loss', 'Validation Loss'])
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[1].plot(epoch_count, training_accuracy, 'r--')
    axs[1].plot(epoch_count, validation_accuracy, 'b-')
    axs[1].legend(['Training Accuracy', 'Validation Accuracy'])
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    fig.savefig('VGG19_1.png')

    #plt.plot(his.history['accuracy'], label='train')
    #plt.plot(his.history['val_accuracy'], label='test')
    #plt.title('lrate='+str(lrate), pad=-50)


    return model

def main(weights_file):

    # Fine tune pretrained VGG19
    model = get_model()

    # Train vgg19 model from scratch
    #model = vgg16()
    #print("Number of layers in the base model: ", len(model.layers))
    generators = get_generators()

    if weights_file is None:
        print("Loading network from ImageNet weights.")
        # Get and train the top layers.
        model = freeze_all_but_top(model)
        '''
        learning_rates = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]
        for i in range(len(learning_rates)):
            plot_no = 420 + (i+1)
            plt.subplot(plot_no)
            model = train_model(model, 200, generators, learning_rates[i],[early_stopper])'''
        model = train_model(model, 300, generators,
                        [checkpointer, early_stopper, tensorboard, csv_logger])
        #plt.savefig('VGG19_Learning_Rate.png',bbox_inches = "tight")
        #plt.close() 
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    


if __name__ == '__main__':
    weights_file = None
    #print(len(data.classes))
    main(weights_file)
