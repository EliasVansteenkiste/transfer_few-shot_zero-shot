#!/usr/bin/env bash

export LC_ALL=C
apt-get update
apt-get -y install htop nano libgtk2.0-dev screen libvips
python3 -m pip install --upgrade pip
python3 -m pip install pytest
python3 -m pip install torch==1.4.0 sacred==0.7.4 tqdm==4.45.0 pytorch-ignite==0.2.1 scikit-learn==0.22.2.post1
python3 -m pip install tensorboardX==2.0 imgaug==0.3.0 kornia==0.2.0 opencv-python
