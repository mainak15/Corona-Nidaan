
from keras.applications.vgg19 import VGG19
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input,concatenate,add
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization,SeparableConv2D
from keras.layers.convolutional import Conv2D, MaxPooling2D,AveragePooling2D
import time
import os.path
import itertools
import cv2
from glob import glob
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD ,RMSprop
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
import math 
import keras
from tensorflow.keras.regularizers import l2

#data = DataSet()
train_data_dir = 'data/train3'
valid_data_dir = 'data/test3'
# Helper: Save the model.
batch_size=8
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'covid_vs_normal_vs_pneumonia_v1_2_1.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
#early_stopper = EarlyStopping(patience=10)
early_stopper = EarlyStopping(monitor='val_loss', patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('data', 'logs', 'covid_vs_normal_vs_pneumonia_v1' + '-' + 'training-' + \
        str(timestamp) + '.log'))

def multiple_outputs(generator, image_dir, batch_size, image_size,classes):
    gen = generator.flow_from_directory(
        image_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical')
    
    while True:
        gnext = gen.next()
        # return image batch and 3 sets of lables
        yield gnext[0], [gnext[1], gnext[1], gnext[1]]

def get_generators():
    '''
    train_datagen = ImageDataGenerator(
        rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
        rotation_range=10,
                )'''

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        shuffle=True,
        classes=['COVID-19','normal','pneumonia'],
        class_mode='categorical')
    '''
    train_generator = multiple_outputs(
        train_datagen,
        image_dir=train_data_dir,
        batch_size=4,
        image_size=256,
        classes=['COVID-19','normal','pneumonia'])
    '''
    validation_generator = test_datagen.flow_from_directory(
        valid_data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        shuffle=True,
        classes=['COVID-19','normal','pneumonia'],
        class_mode='categorical')
    '''
    validation_generator = multiple_outputs(
        test_datagen,
        image_dir=valid_data_dir,
        batch_size=4,
        image_size=256,
        classes=['COVID-19','normal','pneumonia'])'''

    return train_generator, validation_generator

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None,n_filters=None):
    
    conv_1x1 = SeparableConv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_1x1 = SeparableConv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_1x1)
    conv_1x1 = BatchNormalization()(conv_1x1)

    
    conv_3x3 = SeparableConv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = BatchNormalization()(conv_3x3)
    conv_3x3 = SeparableConv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)
    conv_3x3 = BatchNormalization()(conv_3x3)
    conv_3x3 = SeparableConv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)
    conv_3x3 = BatchNormalization()(conv_3x3)

    conv_5x5 = SeparableConv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = BatchNormalization()(conv_5x5)
    conv_5x5 = SeparableConv2D(filters_5x5, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)
    conv_5x5 = BatchNormalization()(conv_5x5)
    conv_5x5 = SeparableConv2D(filters_5x5, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)
    conv_5x5 = BatchNormalization()(conv_5x5)
    conv_5x5 = SeparableConv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)
    conv_5x5 = BatchNormalization()(conv_5x5)

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = SeparableConv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)
    pool_proj = BatchNormalization()(pool_proj)
    pool_proj = SeparableConv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)
    pool_proj = BatchNormalization()(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

def residual_module(layer_in,filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     n_filters=None,name=None):
    merge_input = layer_in
    print(layer_in.shape[-1])
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        print('hiiiii')
        merge_input = SeparableConv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(layer_in)
    x = inception_module(layer_in,
                         filters_1x1=filters_1x1,
                         filters_3x3_reduce=filters_3x3_reduce,
                         filters_3x3=filters_3x3,
                         filters_5x5_reduce=filters_5x5_reduce,
                         filters_5x5=filters_5x5,
                         filters_pool_proj=filters_pool_proj,
                         name=name,n_filters=int(layer_in.shape[-1]))
    x = SeparableConv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    # add filters, assumes filters/channels last
    layer_out = add([x, merge_input])
    # activation function
    layer_out = Activation('relu')(layer_out)
    return layer_out

epochs = 25
initial_lrate = 0.01
def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)

lr_sc = LearningRateScheduler(decay, verbose=1)
'''
optimizer = Adam(lr=1e-4, decay=1e-3)
scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.7,
                                  patience=5,
                                  mode='max',
                                  min_lr=1e-7)'''

def srima():
    input_layer = Input(shape=(256, 256, 3))

    x = SeparableConv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, name='conv_1_7x7/2')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    #x = SeparableConv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    #x = BatchNormalization()(x)
    x = SeparableConv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, name='conv_2b_3x3/1')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x=residual_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         n_filters=256,
                         name='inception_3a')
    x=residual_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         n_filters=480,
                         name='inception_3b')
    #x = BatchNormalization()(x)
    #x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    x=residual_module(x,
                         filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                         n_filters=832,
                         name='inception_4a')

    '''
    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')
    #x = BatchNormalization()(x)
    #x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    #x=residual_module(x, 64)
    
    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         name='inception_3b')

    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)


    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         name='inception_4a')'''

    x1 = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
    
    #x1 = SeparableConv2D(128, (1, 1), padding='same', activation='relu')(x1)
    #x1 = BatchNormalization()(x1)
    #x1 = Flatten()(x1)
    #x1 =GlobalAveragePooling2D()(x1)
    
    #x1 = Dense(1024, activation='relu',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(x1)
    x1 = Dropout(0.4)(x1)
    x1 = Dense(3, activation='softmax', name='auxilliary_output_1')(x1)



    model = Model(input_layer, x1, name='COVIDNet')
    
    # compile the model (should be done *after* setting layers to non-trainable)

    #optimizer = Adam(lr=1e-3, decay=1e-5)
    #optimizer=SGD(lr=0.0001, momentum=0.9)
    #model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    #from contextlib import redirect_stdout

    #with open('modelsummary.txt', 'w') as f:
    #    with redirect_stdout(f):
    #        model.summary()    
    #print(model.summary())
    #plot_model(model, to_file='model_plot_covid_vs_normal_vs_pneumonia_v1_2.png', show_shapes=True, show_layer_names=True)
    print("Number of layers in the base model: ", len(model.layers))

    return model
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
#scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3,verbose=1, min_lr=0.00001)
def train_model(model, nb_epoch, generators,callbacks=[]):
    from sklearn.utils import class_weight
    import numpy as np

    train_generator, validation_generator = generators
    
    #optimizer = Adam(lr=0.01)
    #optimizer=RMSprop(lr=0.0001)
    #optimizer=SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes)
    #class_weights = dict(enumerate(class_weights))
    print(class_weights)
    #class_weights={0:25. ,1:1. ,2:1. }
    #print(train_generator.classes)
    his=model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames) // batch_size,
        validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames) // batch_size,
        epochs=nb_epoch,
        class_weight=class_weights,
        #class_weight='balanced',
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
    fig.savefig('covid_vs_normal_vs_pneumonia_v1_2_1.png')
    #plt.plot(his.history['accuracy'], label='train')
    #plt.plot(his.history['val_accuracy'], label='test')
    #plt.title('opt='+optimizer, pad=-80)
    return model

def main(weights_file):

    
    model = srima()
    #print("Number of layers in the base model: ", len(model.layers))
    generators = get_generators()

    if weights_file is None:
        print("Loading network.")
        '''
        momentums = ['sgd', 'rmsprop', 'adagrad', 'adam']
        #learning_rates = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]
        
        #decay_rates = [1E-1, 1E-2, 1E-3, 1E-4]
        for i in range(len(momentums)):
            plot_no = 220 + (i+1)
            plt.subplot(plot_no)
            model = train_model(model, 200, generators, momentums[i],[early_stopper])'''
        model = train_model(model, 300, generators,
                        [checkpointer, early_stopper, tensorboard, csv_logger,learning_rate_reduction])
        #plt.tight_layout()
        #plt.savefig('Model3_Optimizer.png',bbox_inches = "tight")
        #plt.savefig('Model3_Learning_Rate.png',bbox_inches = "tight")
        #plt.savefig('Model3_decay_Rate.png',bbox_inches = "tight")
        #plt.close()
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    


if __name__ == '__main__':
    weights_file = None
    #print(len(data.classes))
    main(weights_file)
