# PytorchComputerVisionFMNIST 
# Overview

**Table of Contents**

Introduction
Dataset
Data Preprocessing
Model Architecture
Training
Model Evaluation
Model Saving and Loading

Welcome to the FashionMNIST Computer Vision Model Documentation! 
This document provides an in-depth overview of the steps and functions used to build, train, evaluate, and save a computer vision model for classifying fashion items in the FashionMNIST dataset.

First we begin with loading the Packages and the dataset. There are manu computer vision datasets available in the torchvision dataset

**Data Preparation**
The FashionMNIST dataset is used for training and testing the model.
The dataset is divided into training and testing subsets.
Each image in the dataset is preprocessed to convert it into a Torch tensor.
The dataset contains grayscale images of fashion items, each with a corresponding label.
Data Loading
DataLoader objects are used to create batches of training and testing data.
The batch size is set using the BATCH_SIZE hyperparameter.
Model Architecture

**Model Creation**
Three different model architectures are implemented:
FashionMNISTModelV0: A simple feedforward neural network with linear layers.
FashionMNISTModelV1: A feedforward neural network with additional non-linear activation functions (ReLU).
FashionMNISTModelV2: A convolutional neural network (CNN) inspired by TinyVGG.

**Model Training**
The models are trained using stochastic gradient descent (SGD) as the optimizer.
Cross-entropy loss is used as the loss function.
Training is performed for a specified number of epochs.
Training and testing loops are implemented to evaluate the model's performance.
Training

Training involves forward and backward passes for each batch of data.
The loss is computed and used to update the model's weights.
The training process is timed using the timeit library.
Model Evaluation

The trained models are evaluated on the **testing dataset**.
Evaluation metrics include loss and accuracy.
A confusion matrix is generated to visualize classification results.
The torchmetrics and mlxtend libraries are used for evaluation.
Model Saving and Loading

Trained models can be saved and loaded for future use.
Model state dictionaries are saved to disk for persistence.
The saved model can be loaded and evaluated on new data.

-- References: I learned and praticed this computer vision model using Freecodecamp's Pytorch lession https://www.youtube.com/watch?v=V_xro1bcAuA&t=205s --
