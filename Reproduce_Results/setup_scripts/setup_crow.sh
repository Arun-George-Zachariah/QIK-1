#!/usr/bin/env bash

# Constants.
HOME=${HOME}
QIK_HOME=${PWD}/../..

# Obtaining the model.
wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel -O $QIK_HOME/ML_Models/CroW/vgg/VGG_ILSVRC_16_layers.caffemodel

# Activating the conda environment.
source activate deepvision

# Extracting the features.
cd $QIK_HOME/ML_Models/CroW && python extract_features.py