"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.

Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""
from keras.applications.densenet import DenseNet201
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
import time
import os.path
import itertools
import cv2
from glob import glob
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
import math 
import keras

#data = DataSet()
train_data_dir = 'data/train3'
valid_data_dir = 'data/test3'
# Helper: Save the model.
batch_size=8
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'densenet.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
#early_stopper = EarlyStopping(patience=10)
early_stopper = EarlyStopping(monitor='val_loss', patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('data', 'logs', 'densenet' + '-' + 'training-' + \
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
    base_model = DenseNet201(weights=weights, include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    print("Number of layers in the base model: ", len(base_model.layers))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(3, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    print(model.summary())
    return model
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[481:]:
        layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    #optimizer = Adam(lr=1e-4, decay=1e-5)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #for layer in model.layers:
    #    print(layer, layer.trainable)

    return model

def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    from sklearn.utils import class_weight
    import numpy as np
    class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes)
    print(class_weights)    
    hist=model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames) // batch_size,
        validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames) // batch_size,
        epochs=nb_epoch,
        class_weight=class_weights,
        callbacks=callbacks)
    fig, axs = plt.subplots(1, 2, figsize = (15, 4))
    training_loss = hist.history['loss']
    validation_loss = hist.history['val_loss']
    training_accuracy = hist.history['accuracy']
    validation_accuracy = hist.history['val_accuracy']
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
    fig.savefig('DenseNet201.png')
    return model

def main(weights_file):
    model = get_model()
    generators = get_generators()

    if weights_file is None:
        print("Loading network from ImageNet weights.")
        # Get and train the top layers.
        model = freeze_all_but_top(model)
        #model = train_model(model, 300, generators,
        #                [checkpointer, early_stopper, tensorboard, csv_logger,learning_rate_reduction])
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    # Get and train the mid layers.
    


if __name__ == '__main__':
    weights_file = None
    #print(len(data.classes))
    main(weights_file)
