from globals import *
from Agents import Agent, to_variable, to_numpy


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


class A3CNet(nn.Module):
    def __init__(self, config, in_channels=1, num_actions=2):
        super(A3CNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.is_LSTM_mode = config['name'] == 'LSTM'

        if self.is_LSTM_mode:
            self.lstm = nn.LSTMCell(config['hidden_size'], config['hidden_size'])
            self.fc1 = nn.Linear(32 * 9 * 9, config['hidden_size'])
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
        else:
            self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
            self.fc1 = nn.Linear(64 * 7 * 7, config['hidden_size'])

        self.critic_linear = nn.Linear(config['hidden_size'], 1)
        self.actor_linear = nn.Linear(config['hidden_size'], num_actions)

        # Initialize weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, x):  # Compute the network output or Q value and V value,
        if self.is_LSTM_mode:
            inputs, (hx, cx) = x
        else:
            inputs = x

        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))

        if self.is_LSTM_mode:
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            hx, cx = self.lstm(x, (hx, cx))
            x = hx
        else:
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = self.fc1(x)

        if self.is_LSTM_mode:
            result = self.critic_linear(x), self.actor_linear(x), (hx, cx)
        else:
            result = self.critic_linear(x), self.actor_linear(x)

        return result


class A3C(Agent):
    def __init__(self, shared_net, num_actions, in_channels, config):
        super(A3C, self).__init__()
        net = A3CNet(config, in_channels, num_actions)

        self.net = net
        self.config = config

        if config['name'] == 'LSTM':
            self.cx = Variable(torch.zeros(1, config['hidden_size']))
            self.hx = Variable(torch.zeros(1, config['hidden_size']))

        self.shared_net = shared_net

        self.optimizer = torch.optim.Adam(self.shared_net.parameters(), lr=config['lr'])
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.clip_norm = config['clip_norm']
        self.entropy_beta = config['entropy_beta']

    def select_action(self, state, test=False):
        state = to_variable(state, isCuda=False, volatile=test)

        if self.config['name'] == 'LSTM':
            on_state = state, (self.hx, self.cx)
            value, logit, (hx, cx) = self.net.forward(on_state)
        else:
            on_state = state
            value, logit = self.net.forward(on_state)

        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)

        action = prob.multinomial().data
        log_prob = log_prob.gather(1, Variable(action))

        action = action.numpy()[0, 0]

        if self.config['name'] == 'LSTM':
            self.hx, self.cx = hx, cx

        return action, value, log_prob, entropy

    def update(self, values, log_probs, rewards, entropies, done, state=None):

        if done:
            if self.config['name'] == 'LSTM':
                self.cx = Variable(torch.zeros(1, self.config['hidden_size']))
                self.hx = Variable(torch.zeros(1, self.config['hidden_size']))
            R = Variable(torch.zeros(1, 1))
        else:
            if self.config['name'] == 'LSTM':
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            _, value, _, _ = self.select_action(state)
            R = Variable(value.data)

        values.append(R)
        # print(len(values), len(log_probs), len(rewards), len(entropies))

        policy_loss = 0.
        value_loss = 0.

        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]

            delta_t = rewards[i] + self.gamma * values[i + 1].data - values[i].data
            gae = gae * self.gamma * self.tau + delta_t

            policy_loss -= log_probs[i] * Variable(gae) + self.entropy_beta * entropies[i]
            value_loss += 0.5 * (R - values[i]) ** 2

        # Perform asynchronous update
        final_loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()

        final_loss.backward()

        nn.utils.clip_grad_norm(self.net.parameters(), self.clip_norm)

        ensure_shared_grads(self.net, self.shared_net)

        self.optimizer.step()

        # Sync with shared model
        self.net.load_state_dict(self.shared_net.state_dict())

