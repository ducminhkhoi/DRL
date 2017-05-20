from __future__ import division
from collections import deque
import gym
import numpy as np
from PIL import Image
import random

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


viz = Visdom()

np.random.seed(3)
isGPU = torch.cuda.is_available()


is_on_server = socket.gethostname().endswith('eecs.oregonstate.edu')


def plot_reward(plot, list_reward):
    viz.updateTrace(X=np.arange(len(list_reward)),
                        Y=np.array(list_reward),
                        win=plot, append=False)


class Agent:
    def __init__(self):
        super(Agent, self).__init__()

    def preprocess(self, state, volatile=False):
        tensor_state = state.float().unsqueeze(0)

        if isGPU:
            tensor_state = tensor_state.cuda()

        return Variable(tensor_state, volatile=volatile)

    def select_action(self, epoch, state):
        raise NotImplementedError

    def update(self, state, action, reward, new_state, done):
        raise NotImplementedError

    def save(self, file_path):
        torch.save(self.net.state_dict(), file_path)
        print("save model to file successful")

    def load(self, file_path):
        state_dict = torch.load(file_path)
        self.net.load_state_dict(state_dict)
        print("load model to file successful")




