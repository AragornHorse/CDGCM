# CDGCM: A Cheap Dynamic Gesture Controled Video Framework

## What's CDGCV
  CDGCV is a cheap dynamic-gesture-controled-video framework. We provide several reasonable models writen in pytorch, as well as their training code. You can convinently download them for your own projects or just to enjoy our application.

## What're the models used to classify gestures
* ResNet-LSTM
* ResNet-Attention
* ResNet-Bert
* ResNet
* LSTM2D
* ResNet3D

## What else
* A ResNet-LSTM to recognize whether a gesture happened
* A ResNet to recognize whether the user appears
* A fft module to reduce noise
* Several filters to reduce noise

## What're key-points
* Adapting tiny models like ResNet3d, which can be run in most equipments (Although we may loss a little accuracy)
* Adapting tiny data format, using smaller, fewer, grayer images to discribe a gesture-video (Although we may loss a little accuracy)
* Adapting cheap computations like face-recognization to determine whether the classifier is to be run

## What're going to be updated sooner
* The working code to control a video player online
* Neater project format
* Pretrained parameters
* A setup api
* Dataset urls
* More details about our project

