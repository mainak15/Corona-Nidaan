Corona-Nidaan: Lightweight Deep Convolutional Neural Network for Chest X-Ray based COVID-19 Infection Detection.
===========
This research is supported by the Defence Institute of Advanced Technology, DRDO Lab, Ministry of Defence, India, and the Indian National Academy of Engineering. The authors would like to thank Sardar Vallabhbhai Patel COVID Hospital, New Delhi, India, to facilitate validation of Corona-Nidaan against the Indian Patient X-Ray samples. The authors would like to thank NVIDIA for the GPU grant for carrying out deep learning based research work.
<p align="center">
	<img src="images/GUI1.png"  width="100%" height="100%">
	<br>
	<em>Graphical abstract of our proposed Corona-Nidaan model.</em>
</p>
Please see our paper for the details.


## Requirements

The main requirements are listed below:

* Tested with NVIDIA GeForce GTX 1050 Ti
* Tested with CUDA v10.0.130 Tool kit and CuDNN v7.6.5
* Python 3.7.7
* OpenCV 4.1.1
* Keras 2.2.4 API
* Tested with TensorFlow-GPU v1.14.0
* Numpy
* Pillow
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
## Steps for testing
* Locate the checkpoint files (./data/checkpoints/)
* To test the Corona-Nidaan deep neural network or transfer learning model run the following command (provide the name of the checkpoint file to "save_path" in the "covid_vs_normal_vs_pneumonia_prediction.py" script to load the respective trained model and change the "batch_size" as well, it will be the same as training): 
```
python covid_vs_normal_vs_pneumonia_prediction.py
```

## Steps for running the demo app
* Locate the checkpoint files (./data/checkpoints/)
* Provide the name of the checkpoint file to "save_path" in the "covid_vs_normal_vs_pneumonia_prediction.py" script to load the respective trained model and then run the following command:
```
python GUI.py
```





If you find our work useful, can cite our paper using:
```
@article{chakraborty2021corona,
  title={Corona-Nidaan: lightweight deep convolutional neural network for chest X-Ray based COVID-19 infection detection},
  author={Chakraborty, Mainak and Dhavale, Sunita Vikrant and Ingole, Jitendra},
  journal={Applied Intelligence},
  volume={51},
  number={5},
  pages={3026--3043},
  year={2021},
  publisher={Springer}
}
```        
