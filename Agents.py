
from globals import *
from utils import *


class Agent:
    def __init__(self):
        super(Agent, self).__init__()

    def select_action(self, epoch, state):
        raise NotImplementedError

    def update(self, state, action, reward, new_state, done):
        raise NotImplementedError

    def save(self, file_path):
        torch.save(self.net.state_dict(), file_path)
        print("save model to file successful")

    def load(self, file_path):
        state_dict = torch.load(file_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict)
        print("load model to file successful")

from agents.A3C import A3C
from agents.DDPG import DDPG
from agents.DQN import DQN