from globals import *
from Agents import Agent, to_numpy, to_variable
from memory import SequentialMemory


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class OUProcess(object):
    def __init__(self, theta, mu=0., sigma=1., x0=0., dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0., self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x0 = x
        return x


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, use_conv=True, hidden1=20, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        if use_conv:
            self.conv1 = nn.Conv2d(nb_states, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.fc1 = nn.Linear(7 * 7 * 64, hidden1)
        else:
            self.fc1 = nn.Linear(nb_states, hidden1)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.init_weights(init_w)
        self.use_conv = use_conv

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        if self.use_conv:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        y = F.tanh(x)

        return y


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, use_conv=True, hidden1=20, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        if use_conv:
            self.conv1 = nn.Conv2d(nb_states, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.fc1 = nn.Linear(7 * 7 * 64, hidden1)
        else:
            self.fc1 = nn.Linear(nb_states, hidden1)

        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.init_weights(init_w)
        self.use_conv = use_conv

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        if self.use_conv:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(torch.cat([x, a], 1)))
        out = self.fc3(x)
        return out


class DDPG(Agent):
    def __init__(self, in_channels, num_actions, use_conv, config):
        super(DDPG, self).__init__()

        self.nb_states = in_channels
        self.nb_actions = num_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': config['hidden1'],
            'hidden2': config['hidden2'],
            'init_w': config['init_w'],
        }

        self.loss = nn.MSELoss()
        self.actor = Actor(self.nb_states, self.nb_actions, use_conv, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, use_conv, **net_cfg)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config['plr'])

        self.critic = Critic(self.nb_states, self.nb_actions, use_conv, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, use_conv, **net_cfg)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

        if isGPU:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        self.observation = config['observation']
        self.config = config
        self.action_type = config['action_type']

        self.experience_replay = deque(maxlen=config['memory_size'])  # Create Buffer replay

        self.noise = OUProcess(size=self.nb_actions, theta=config['ou_theta'], mu=config['ou_mu'],
                               sigma=config['ou_sigma'])

        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.discount = config['discount']
        self.use_expect = config['use_expect']
        self.pmax = 1
        self.pmin = -1

    def select_action(self, state, test=False):
        value = to_numpy(self.actor.forward(to_variable(state, volatile=True)))

        cur_episode = len(self.experience_replay)

        action = np.clip(value[0] + self.noise.generate(cur_episode), -1, 1)

        return action

    def update(self, state, action, reward, new_state, done):

        self.experience_replay.append((state, action, reward, new_state, done))

        if len(self.experience_replay) >= self.observation:  # if have enough experience example, go
            # Sample batch from memory replay

            mini_batch = random.sample(self.experience_replay, self.batch_size)
            state_batch = torch.cat([mini_batch[k][0].unsqueeze(0) for k in range(self.batch_size)])
            action_batch = [mini_batch[k][1] for k in range(self.batch_size)]
            reward_batch = [mini_batch[k][2] for k in range(self.batch_size)]
            next_state_batch = torch.cat([mini_batch[k][3].unsqueeze(0) for k in range(self.batch_size)])
            terminal_batch = [mini_batch[k][4] for k in range(self.batch_size)]

            action_tensor = to_variable(np.vstack(action_batch))

            # Prepare for the target q batch
            value = self.actor_target.forward(to_variable(next_state_batch, volatile=True))
            next_q_values = self.critic_target.forward([to_variable(next_state_batch, volatile=True), value])

            next_q_values.volatile = False

            try:
                y_batch = to_variable(reward_batch) + self.discount * \
                    to_variable(terminal_batch) * next_q_values
            except RuntimeError:
                print(reward_batch)

            # Critic update
            self.critic.zero_grad()

            q_batch = self.critic.forward([to_variable(state_batch), action_tensor])

            value_loss = self.loss(q_batch, y_batch)
            value_loss.backward()
            self.critic_optim.step()

            # Actor update
            self.actor.zero_grad()

            value = self.actor.forward(to_variable(state_batch))
            policy_loss = -self.critic.forward([to_variable(state_batch), value])

            policy_loss = policy_loss.mean()
            policy_loss.backward()

            self.actor_optim.step()

            # Target update
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

    def save(self, file_path):
        torch.save((self.actor.state_dict(), self.critic.state_dict()), file_path)
        print("save model to file successful")

    def load(self, file_path):
        state_dicts = torch.load(file_path, map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(state_dicts[0])
        self.critic.load_state_dict(state_dicts[1])
        print("load model to file successful")
