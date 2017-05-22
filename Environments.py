from globals import *


def FB_transform(x):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(lambda x1: x1.crop((0, 0, 288, 407))),
        transforms.Lambda(lambda x1: x1.convert('L')),
        transforms.Scale(size=(80, 80)),
        transforms.ToTensor(),
    ])

    return transform(x)


env_transform = {
    'FlappyBird': FB_transform
}