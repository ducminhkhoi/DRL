from globals import *


def plot_reward(plot, list_reward, env):
    viz.updateTrace(X=np.arange(len(list_reward)),
                        Y=np.array(list_reward),
                        win=plot, append=False, env=env)



