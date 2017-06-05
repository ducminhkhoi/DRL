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


class GreedyPolicy(object):
    def __init__(self, action_dim, n_steps_annealing, min_epsilon, max_epsilon):
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.action_dim = action_dim
        self.n_steps_annealing = n_steps_annealing
        self.epsilon_step = - (self.epsilon - self.min_epsilon) / float(self.n_steps_annealing)

    def generate(self, action, step):
        epsilon = max(self.min_epsilon, self.epsilon_step * step + self.epsilon)

        if random.random() < epsilon:
            return random.choice(range(self.action_dim))
        else:
            return action


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, action_type, use_expect, hidden1=20, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(nb_states, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.init_weights(init_w)
        self.action_type = action_type

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.action_type == 'continuous':
            x = F.tanh(x)
        else:
            x = F.softmax(x)
            x = x

        return x


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, action_type, use_expect, hidden1=20, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(nb_states, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, hidden1)
        self.fc2 = nn.Linear(hidden1+1, hidden2) if use_expect else nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(torch.cat([x, a], 1)))
        out = self.fc3(x)
        return out


class DDPG(Agent):
    def __init__(self, in_channels, num_actions, config):
        super(DDPG, self).__init__()

        self.nb_states = in_channels
        self.nb_actions = num_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': config['hidden1'],
            'hidden2': config['hidden2'],
            'init_w': config['init_w'],
            'use_expect': config['use_expect'],
            'action_type': config['action_type']
        }

        self.loss = nn.MSELoss()
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config['plr'])

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
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

        if self.action_type == 'continuous':
            self.noise = OUProcess(size=self.nb_actions, theta=config['ou_theta'], mu=config['ou_mu'],
                                        sigma=config['ou_sigma'])
        else:
            self.noise = GreedyPolicy(action_dim=self.nb_actions, n_steps_annealing=config['epsilon_decay'],
                                               min_epsilon=config['min_epsilon'], max_epsilon=config['max_epsilon'])

        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.discount = config['discount']
        self.use_expect = config['use_expect']
        self.pmax = 1
        self.pmin = -1

    def select_action(self, state, test=False):
        value = to_numpy(self.actor.forward(to_variable(state, volatile=True)))
        # print(value)

        cur_episode = len(self.experience_replay)

        if self.action_type == 'continuous':
            action = np.clip(value[0] + self.noise.generate(cur_episode), -1, 1)
        else:
            action = self.noise.generate(value[0], cur_episode)
            if isinstance(action, int):
                action = np.array([1., 0.] if action == 0 else [0., 1.])
            else:
                # action = np.clip(action, 0.4, 0.6)
                action = action

        return action

    def compute_expect(self, state, value, network, volatile=False):

        if self.use_expect:

            actions = [to_variable(i * torch.ones(value.size(0), 1)) for i in range(self.nb_actions)]

            next_q_values = torch.cat([network.forward([to_variable(state, volatile=volatile), a])
                                       for a in actions], 1)

            next_q_values = torch.cat([value[i, :].dot(next_q_values[i, :])
                                       for i in range(self.batch_size)], 0)
        else:
            next_q_values = network.forward([to_variable(state, volatile=volatile), value])

        return next_q_values

    def invert_gradient(self, d, g):
        if len(d.size()) == 1:
            for i in range(d.size(0)):
                if g[i] > 0:
                    g[i] *= abs(self.pmax - d[i]) / (self.pmax - self.pmin)
                else:
                    g[i] *= abs(d[i] - self.pmin) / (self.pmax - self.pmin)
        else:
            for d_, g_ in zip(d, g):
                self.invert_gradient(d_, g_)

    def update(self, state, action, reward, new_state, done):

        self.experience_replay.append((state, action, reward, new_state, done))

        if len(self.experience_replay) >= self.observation:  # if have enough experience example, go
            # Sample batch from memory replay

            mini_batch = random.sample(self.experience_replay, self.batch_size)
            state_batch = torch.cat(mini_batch[k][0].unsqueeze(0) for k in range(self.batch_size))
            action_batch = [mini_batch[k][1] for k in range(self.batch_size)]
            reward_batch = [mini_batch[k][2] for k in range(self.batch_size)]
            next_state_batch = torch.cat(mini_batch[k][3].unsqueeze(0) for k in range(self.batch_size))
            terminal_batch = [mini_batch[k][4] for k in range(self.batch_size)]

            action_tensor = to_variable(np.vstack(action_batch))

            # Prepare for the target q batch
            value = self.actor_target.forward(to_variable(next_state_batch, volatile=True))
            next_q_values = self.compute_expect(next_state_batch, value, self.critic_target, volatile=True)

            next_q_values.volatile = False

            y_batch = to_variable(reward_batch) + self.discount * \
                to_variable(terminal_batch) * next_q_values

            # Critic update
            self.critic.zero_grad()

            q_batch = self.compute_expect(state_batch, action_tensor, self.critic)
            # q_batch = self.critic.forward([to_variable(state_batch), action_tensor])

            value_loss = self.loss(q_batch, y_batch)
            value_loss.backward()
            self.critic_optim.step()

            # Actor update
            self.actor.zero_grad()

            value = self.actor.forward(to_variable(state_batch))
            policy_loss = -self.compute_expect(state_batch, value, self.critic)
            # policy_loss = -self.critic.forward([to_variable(state_batch), value])

            policy_loss = policy_loss.mean()
            policy_loss.backward()

            # torch.nn.utils.clip_grad_norm(self.actor.parameters(), 1.)

            list_params = list(self.actor.parameters())
            # print(list_params[-1].grad[0])

            # # invert gradients
            # for i, p in enumerate(list_params):
            #     if i == len(list_params)-1:
            #         for j in range(self.nb_actions):
            #             # print("gradient", p.grad.data[j])
            #             if p.grad.data[j] > 0:  # suggest increasing p
            #                 # print("current p", (self.pmax - p.data[j]) / (self.pmax - self.pmin))
            #                 p.grad.data[j] *= abs(self.pmax - p.data[j])/(self.pmax - self.pmin)
            #             else:
            #                 # print("current p", (p.data[j] - self.pmin) / (self.pmax - self.pmin))
            #                 p.grad.data[j] *= abs(p.data[j] - self.pmin)/(self.pmax - self.pmin)

            # for p in list_params:
            #     self.invert_gradient(p.data, p.grad.data)

            # print(list_params[-1].grad[0])

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
