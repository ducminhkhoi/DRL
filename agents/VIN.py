from globals import *
from Agents import Agent


class VINNet(nn.Module):
    def __init__(self, in_channels=4, num_actions=2):
        super(VINNet, self).__init__()

    def forward(self, x):
        pass


class VIN(Agent):
    def __init__(self):
        super(VIN, self).__init__()
        net = VINNet()

    def select_action(self, epoch, state):
        pass

    def update(self, state, action, reward, new_state, done):
        pass
