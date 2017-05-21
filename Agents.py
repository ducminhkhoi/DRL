
from globals import *


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

from agents.A3C import A3C
from agents.DDPG import DDPG
from agents.DQN import DQN
from agents.ES import ES
from agents.VIN import VIN