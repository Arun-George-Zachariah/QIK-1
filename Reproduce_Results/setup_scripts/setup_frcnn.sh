#!/usr/bin/env bash

# Constants.
HOME=${HOME}
QIK_HOME=${PWD}/../..

# Installing FR-CNN.
cd $QIK_HOME/ML_Models/DeepVision && ./setup.sh

echo "After FRCNN execution : $(pwd)"

# Extracting features.
python read_data.py && python features.py

# Starting the FR-CNN web app.
python qik_search.py &>> /dev/null &

