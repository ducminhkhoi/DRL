from globals import *


class DDPG(Agent):
    def __init__(self):
        super(DDPG, self).__init__()

    def select_action(self, epoch, state):
        pass

    def update(self, state, action, reward, new_state, done):
        pass
