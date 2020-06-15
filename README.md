Corona-Nidaan: Lightweight Deep Convolutional Neural Network for Chest X-Ray based COVID-19 Infection Detection.
===========




## Requirements

The main requirements are listed below:

* Tested with NVIDIA GeForce GTX 1050 Ti
* Tested with CUDA v10.0.130 Tool kit and CuDNN v7.6.5
* Python 3.7.7
* OpenCV 4.1.1
* Keras 2.2.4 API
* Tested with TensorFlow-GPU v1.14.0
* Numpy
* Scikit-Learn
* Matplotlib



## ChestX Dataset
The dataset was formed by combining three different open access chest X-ray datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://github.com/agchung/Figure1-COVID-chestxray-dataset
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

## Steps for training
* To train the Corona-Nidaan deep neural network from scratch using the ChestX dataset run the following command:
```
python model_covid_vs_normal_vs_pneumonia_v1.py
```
* To train the transfer learning model run the following commands:
```
python train_DenseNet201.py
python train_InceptionResNetV2.py
python train_InceptionV3.py
python train_MobileNetV2.py
python train_VGG19.py
```








If you find our work useful, can cite our paper using:
```
@misc{
        }
```        
