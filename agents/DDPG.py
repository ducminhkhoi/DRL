from globals import *
from Agents import Agent


class DDPGNet(nn.Module):
    def __init__(self, in_channels=4, num_actions=2):
        super(DDPGNet, self).__init__()

    def forward(self, x):
        pass


class DDPG(Agent):
    def __init__(self):
        super(DDPG, self).__init__()
        net = DDPGNet()

    def select_action(self, epoch, state):
        pass

    def update(self, state, action, reward, new_state, done):
        pass
