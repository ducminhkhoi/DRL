from __future__ import division
from collections import deque
import gym
import numpy as np
from PIL import Image
import random
from time import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from torchvision import transforms
from visdom import Visdom
import gym
import gym_ple
import pygame
import socket
from torch.nn import init
from copy import deepcopy
from torch.nn.parameter import Parameter

viz = Visdom()

np.random.seed(3)
isGPU = torch.cuda.is_available()


is_on_server = socket.gethostname().endswith('eecs.oregonstate.edu')






