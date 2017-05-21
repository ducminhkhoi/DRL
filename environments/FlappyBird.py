from globals import *
from utils import *
from Environments import Environment


class FlappyBirdNet(Environment):
    def __init__(self, in_channels=1, num_actions=2, using_rnn=True):
        super(FlappyBirdNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(6 * 6 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        self.use_rnn = using_rnn
        self.in_channels = in_channels
        if using_rnn:
            self.rnn = nn.RNN(512, 512, 1)

        # weight initialize
        weights_init([self.conv1, self.conv2, self.conv3, self.fc4, self.fc5], "normal")

    def cnn(self, x):
        x = x.contiguous().view(-1, self.in_channels, 80, 80)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        return x

    def forward(self, x):  # Compute the network output or Q value
        if self.use_rnn:
            x = x.view(x.size(0), -1, self.in_channels, x.size(2), x.size(3))
            if isGPU:
                h0 = Variable(torch.zeros(1, x.size(0), 512).cuda(), requires_grad=False)
            else:
                h0 = Variable(torch.zeros(1, x.size(0), 512), requires_grad=False)

            n = torch.cat([self.cnn(x[:, i, :, :, :]) for i in range(x.size(1))], 0)
            cnn_outputs = n.view(x.size(1), x.size(0), n.size(1))
            rnn_outputs = self.rnn(cnn_outputs, h0)
            x = rnn_outputs[0][-1, :, :]
        else:
            x = self.cnn(x)
        return self.fc5(x)

    def transform(self, x):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x1: x1.crop((0, 0, 288, 407))),
            transforms.Lambda(lambda x1: x1.convert('L') if self.in_channels != 3 else x1),
            transforms.Scale(size=(80, 80)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

        return transform(x)
