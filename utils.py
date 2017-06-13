from globals import *


def plot_reward(plot, list_reward, env):
    if isinstance(list_reward[0], (list, tuple)):
        length = len(list_reward[0])
        X = np.column_stack(tuple(np.arange(length) for _ in range(len(list_reward))))
        Y = np.array(list_reward).T
        opts = dict(legend=True)
    else:
        length = len(list_reward)
        X = np.arange(length)
        Y = np.array(list_reward)
        opts = dict(legend=False)

    viz.updateTrace(X=X, Y=Y, win=plot, append=False, env=env, opts=opts)


def makeVariable(x, volatile=False):
    x = (x.cuda() if isGPU else x)
    return Variable(x, volatile=volatile)


def to_numpy(var):
    return var.cpu().data.numpy() if isGPU else var.data.numpy()


def to_variable(var, volatile=False, requires_grad=False, dtype='float'):

    if isinstance(var, np.ndarray):
        var = torch.from_numpy(var)
    elif isinstance(var, (list, tuple)):
        var = torch.FloatTensor(var)

    if len(var.size()) in [1, 3]:  # just have one example, unsqueeze(0)
        tensor_var = getattr(var, dtype)().unsqueeze(0)
    else:
        tensor_var = getattr(var, dtype)()

    if isGPU:
        tensor_var = tensor_var.cuda()

    return Variable(tensor_var, volatile=volatile, requires_grad=requires_grad)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(w, t=1.0):
    e = np.exp(w / t)
    dist = e / np.sum(e)
    return dist
