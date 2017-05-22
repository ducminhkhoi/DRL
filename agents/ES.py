from globals import *
from Agents import Agent


class ESNet(nn.Module):
    def __init__(self, in_channels=4, num_actions=2):
        super(ESNet, self).__init__()

    def forward(self, x):
        pass


class ES(Agent):
    def __init__(self):
        super(ES, self).__init__()
        net = ESNet()

    def select_action(self, epoch, state):
        pass

    def update(self, state, action, reward, new_state, done):
        pass
