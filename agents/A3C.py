from globals import *
from Agents import Agent


class A3C(Agent):
    def __init__(self):
        super(A3C, self).__init__()

    def select_action(self, epoch, state):
        pass

    def update(self, state, action, reward, new_state, done):
        pass
