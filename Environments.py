from globals import *


class FlappyBirdNet(nn.Module):
    def __init__(self, in_channels=4, num_actions=2):
        super(FlappyBirdNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(6 * 6 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):  # Compute the network output or Q value
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        return self.fc5(x)

    @staticmethod
    def transform(x):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: x.crop((60, 0, 288, 407))),
            transforms.Lambda(lambda x: x.convert('L')),
            transforms.Scale(size=(80, 80), interpolation=Image.CUBIC),
            transforms.ToTensor(),
        ])

        return transform(x)


# Define other nets for other games
