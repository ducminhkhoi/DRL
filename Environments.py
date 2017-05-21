from utils import *


class Environment(nn.Module):
    def __init__(self, in_channels=4, num_actions=2):
        super(Environment, self).__init__()

    def forward(self, x):  # Compute the network output or Q value
        raise NotImplementedError

    @staticmethod
    def transform(x):
        raise NotImplementedError

from environments.FlappyBird import FlappyBirdNet