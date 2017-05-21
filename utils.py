from globals import *


def plot_reward(plot, list_reward, env):
    viz.updateTrace(X=np.arange(len(list_reward)),
                        Y=np.array(list_reward),
                        win=plot, append=False, env=env)


def weights_init(ms, init_types):
    if not isinstance(ms, (list, tuple)):
        ms = [ms]

    if not isinstance(init_types, (list, tuple)):
        init_types = [init_types for _ in range(len(ms))]

    for m, init_type in zip(ms, init_types):
        getattr(init, init_type)(m.weight.data, std=0.01)
        getattr(init, init_type)(m.bias.data, std=0.01)
