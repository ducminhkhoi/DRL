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


class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OUProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OUProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=20, hidden2=300, hidden3=600, init_w=3e-3):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(nb_states, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(6 * 6 * 32, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        # self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        # self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        d = self.fc3(x)
        c = F.tanh(d)

        return c, d


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=20, hidden2=300, hidden3=600, init_w=3e-3):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(nb_states, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(6 * 6 * 64 + nb_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        # self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc3 = nn.Linear(hidden2, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        # self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(torch.cat([x, a], 1)))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
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
            # 'hidden3': config['hidden3'],
            # 'hidden4': config['hidden4'],
            'init_w': config['init_w']
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

        if config['use_memory']:
            self.experience_replay = SequentialMemory(limit=config['memory_size'], window_length=1)
        else:
            self.experience_replay = deque(maxlen=config['memory_size'])  # Create Buffer replay

        self.random_process = OUProcess(size=self.nb_actions, theta=config['ou_theta'], mu=config['ou_mu'],
                                        sigma=config['ou_sigma'])

        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.discount = config['discount']
        self.depsilon = 1. / config['epsilon_decay']

        self.epsilon = 1.0

    def select_action(self, state, test=False):
        value_c, value_d = self.actor.forward(to_variable(state, volatile=True))

        action_d = (F.softmax(value_d))
        action_d = to_numpy(action_d.multinomial())

        action_c = to_numpy(value_c)
        action_c += (max(self.epsilon, 0) * self.random_process.sample()) if not test else 0
        action_c = action_c[0]
        return action_c, action_d

    def update(self, state, action, reward, new_state, done):
        if self.config['use_memory']:
            self.experience_replay.append(
                new_state.numpy(), action.tolist(), reward, done)  # add new transition to dataset
        else:
            self.experience_replay.append((state, action.tolist(), reward, new_state, done))

        if done:
            self.random_process.reset_states()

        self.epsilon -= self.depsilon

        if len(self.experience_replay) >= self.observation:  # if have enough experience example, go
            # Sample batch from memory replay

            if self.config['use_memory']:
                state_batch, action_batch, reward_batch, \
                next_state_batch, terminal_batch = self.experience_replay.sample_and_split(self.batch_size)
                state_batch = state_batch.reshape(-1, 4, 80, 80)
                next_state_batch = next_state_batch.reshape(-1, 4, 80, 80)

            else:
                mini_batch = random.sample(self.experience_replay, self.batch_size)
                state_batch = torch.cat(mini_batch[k][0].unsqueeze(0) for k in range(self.batch_size))
                action_batch = [mini_batch[k][1] for k in range(self.batch_size)]
                reward_batch = [mini_batch[k][2] for k in range(self.batch_size)]
                next_state_batch = torch.cat(mini_batch[k][3].unsqueeze(0) for k in range(self.batch_size))
                terminal_batch = [mini_batch[k][4] for k in range(self.batch_size)]

            # Prepare for the target q batch
            value_c, _ = self.actor_target.forward(to_variable(next_state_batch, volatile=True))
            next_q_values = self.critic_target.forward([to_variable(next_state_batch, volatile=True), value_c])
            next_q_values.volatile = False

            y_batch = to_variable(reward_batch) + self.discount * \
                to_variable(terminal_batch) * next_q_values

            # Critic update
            self.critic.zero_grad()

            q_batch = self.critic.forward([to_variable(state_batch), to_variable(action_batch)])

            value_loss = self.loss(q_batch, y_batch)
            value_loss.backward()
            self.critic_optim.step()

            # Actor update
            self.actor.zero_grad()

            value_c, _ = self.actor.forward(to_variable(state_batch))
            policy_loss = -self.critic.forward([to_variable(state_batch), value_c])

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
