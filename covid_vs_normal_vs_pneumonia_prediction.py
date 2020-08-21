
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,load_model
import os.path
import itertools
import cv2
from glob import glob
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from keras.optimizers import SGD


save_path = os.path.join('data', 'checkpoints', 'covid_vs_normal_vs_pneumonia_v1_2_1.hdf5')
#valid_data_dir = 'data/test3'
valid_data_dir = 'data/predict'
batch_size=8

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()
    plt.savefig('covid_vs_normal_vs_pneumonia_v1_2_1_confusion_matrix.png',bbox_inches = "tight")
    plt.close() 

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
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator  = test_datagen.flow_from_directory(
        valid_data_dir,
        target_size=(256, 256),
        batch_size=8,
        shuffle=False,
        class_mode="categorical")
    '''
    validation_generator = multiple_outputs(
        test_datagen,
        image_dir=valid_data_dir,
        batch_size=4,
        image_size=256,
        classes=['COVID-19','normal','pneumonia'])'''

    return  test_generator
epochs = 25
initial_lrate = 0.01

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)
optimizer=SGD(lr=0.0001, momentum=0.9)

def predict_model(generators):
    test_generator = generators
    model = load_model(save_path)
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #print(model.summary())
    #print("Number of layers in the base model: ", len(model.layers))
    print(test_generator.classes)
    predictions = model.predict_generator(
			generator = test_generator,
			workers=1,
			steps = len(test_generator.filenames) // batch_size+1 ,
			verbose = 1
			)
    predictions = np.argmax(predictions, axis=1)
    print(len(predictions))

    labels = (test_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predict = [labels[k] for k in predictions]
    filenames=test_generator.filenames
    print(len(filenames))
    print(len(predict))
    results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predict})
    results.to_csv("results.csv",index=False)
          

    print('Confusion Matrix')
    cm=confusion_matrix(test_generator.classes, predictions)
    plot_confusion_matrix(cm , 
                      normalize    = False,
                      target_names = ['COVID-19','Normal','Pneumonia'],
                      title        = "Confusion Matrix")
    print('Classification Report')
    target_names = ['COVID-19','Normal','Pneumonia']
    print(classification_report(test_generator.classes, predictions, target_names=target_names))

    return model

def main(weights_file):

    # Fine tune pretrained VGG19
    #model = get_model()

    # Train vgg19 model from scratch
    #model = srima()
    #print("Number of layers in the base model: ", len(model.layers))
    generators = get_generators()

    if weights_file is None:
        print("Loading saved model:")
        # Get and train the top layers.
        #model = freeze_all_but_top(model)
        model = predict_model(generators)
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    


if __name__ == '__main__':
    weights_file = None
    #print(len(data.classes))
    main(weights_file)
