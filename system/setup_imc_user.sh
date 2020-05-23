#!/usr/bin/env bash

python3 -m pip install --upgrade pip
python3 -m pip install pytest
python3 -m pip install numpy==1.17 tensorboardX==2.0 imgaug==0.3.0 kornia==0.2.0 opencv-python tqdm==4.45.0
python3 -m pip install torch==1.4.0 torchvision==0.5.0 sacred==0.7.4  pytorch-ignite==0.2.1 scikit-learn==0.22.2.post1