from __future__ import division
import os
os.nice(20)
from collections import deque
import gym
import numpy as np
from PIL import Image
import random
import time
import shutil

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')
from multiprocessing import Manager, Value
from torch.nn import init
from torch.nn.parameter import Parameter

from visdom import Visdom
import gym
import gym_ple
import pygame
import socket
import pickle

from copy import deepcopy
from itertools import product

viz = Visdom()

np.random.seed(3)
isGPU = torch.cuda.is_available()


is_on_server = socket.gethostname().endswith('eecs.oregonstate.edu')
