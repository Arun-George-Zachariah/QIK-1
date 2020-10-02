#!/usr/bin/env bash

# Constants.
HOME=${HOME}
QIK_HOME=${PWD}/../..
QIK_CORE_NAME=QIK
USER=${USER}

# Activating the Conda Environment.
conda create -y --name qik_env python=3.6
conda activate qik_env

# Creating data directories.
cd $QIK_HOME/ML_Models/DeepImageRetrieval && mkdir QIK_Data && cd QIK_Data

# Downloading the DIR Model.
wget https://mailmissouri-my.sharepoint.com/:u:/g/personal/az2z7_mail_umkc_edu/EU5h_fnPLJhBvevls58-EjgBkOvbZYG19DwKlXmfH1eDHg?download=1 -O Resnet-101-AP-GeM.pt

# Copying DIR Image List.
cp $QIK_HOME/Reproduce_Results/data/15K_Dataset.txt DIR_Candidates.txt

# Extracting DIR Features.
cd ../ && python -m dirtorch.extract_features --dataset 'ImageList("QIK_Data/DIR_Candidates.txt")' --checkpoint QIK_Data/Resnet-101-AP-GeM.pt --output QIK_Data/QIK_DIR_Features  --gpu 1