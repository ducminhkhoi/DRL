from globals import *


def FB_transform(x):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(lambda x1: x1.crop((0, 0, 288, 407))),
        transforms.Lambda(lambda x1: x1.convert('L')),
        transforms.Scale(size=(84, 84)),
        transforms.ToTensor(),
    ])

    return transform(x)


def Transform(x):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Lambda(lambda x1: x1.crop((0, 0, 288, 407))),
        transforms.Lambda(lambda x1: x1.convert('L')),
        transforms.Scale(size=(84, 84)),
        transforms.ToTensor(),
    ])

    return transform(x)


env_transform = {
    'FlappyBird-v0': FB_transform,
    'MountainCarContinuous-v0': Transform,
    'MountainCar-v0': Transform,
    'BipedalWalkerHardcore-v2': Transform,
    'BipedalWalker-v2': Transform,
    'CarRacing-v0': Transform,
    'LunarLanderContinuous-v2': Transform,
    'LunarLander-v2': Transform,
}

action_maps = {
    'CarRacing-v0': {
        0: (-1, 0, 0),
        1: (-1, 0, 1),
        2: (-1, 1, 0),
        3: (-1, 1, 1),
        4: (0, 0, 0),
        5: (0, 0, 1),
        6: (0, 1, 0),
        7: (0, 1, 1),
        8: (1, 0, 0),
        9: (1, 0, 1),
        10: (1, 1, 0),
        11: (1, 1, 1),
    },
    'BipedalWalkerHardcore-v2': {i: x for i, x in enumerate(product(*([[-1, 0, 1]]*4)))},
    'BipedalWalker-v2': {i: x for i, x in enumerate(product(*([[-1, 0, 1]]*4)))},
    'FlappyBird-v0': {0: 0, 1: 1},
    'MountainCar-v0': {i: x for i, x in enumerate(range(2))},
    'MountainCarContinuous-v0': {i: x for i, x in enumerate(range(2))},
    'LunarLanderContinuous-v2':  {i: x for i, x in enumerate(product(*([[-1, 0, 1]]*2)))},
    'LunarLander-v2': {i: x for i, x in enumerate(range(4))}
}