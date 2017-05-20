#!/usr/bin/env bash

pip install http://download.pytorch.org/whl/torch-0.1.12.post2-cp36-cp36m-macosx_10_7_x86_64.whl
pip install torchvision

pip install visdom
pip install gym
pip install pygame

git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip install -e .

cd ..
git clone https://github.com/lusob/gym-ple.git
cd gym-ple/
pip install -e .

Xvnc :99 -ac -screen 0 1280x1024x24 &

python -m visdom.server