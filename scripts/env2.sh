#!/bin/bash

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to appropiate environment


cd DiffBIR


conda create -n diffbir python=3.10 -y && conda activate diffbir
pip install -r requirements.txt


cd ..