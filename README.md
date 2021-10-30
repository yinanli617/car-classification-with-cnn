# Car Classification with CNN

## Introduction

This project is about car classification for stanford car dataset. The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

It is difficult to directly train deep learning model on this dataset due to the limited number of images. Thus I use transfer learning, a common approch used in deep learning to utilize the pretrained model on imagenet and fine-tune on our own dataset, i.e. car dataset.

## Results

A step-by-step notebook `stanford-car-classification-efficientnet.ipynb` is provided in this repository.

I use a pretrained `efficientnet_b3` network and find tune it on the car dataset. After 20 epochs of training, the model achieves **88.9% accuracy on the test dataset**, showing the power of transfer learning.

## How to train

There are two ways to train the efficientnet:

1. To train locally
   - Simply clone this repository, and run `python car-cnn.py` (check the python file to look up for available arguments);

2. To train on Kubernetes
   - As is shown in my [prediction of stock price with RNN project](https://github.com/yinanli617/Stock-price-prediction-with-RNN), the training can be deployed on Kubernetes using [the pytorch operator](https://github.com/kubeflow/pytorch-operator);
   - A Dockerfile to build the image is provided in `/docker/`;
   - The pytorch job can be created on Kubernetes by simply running `kubectl apply -f car-cnn-pytorch-job.yaml`