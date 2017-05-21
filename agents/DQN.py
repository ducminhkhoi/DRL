from globals import *
from Agents import Agent


class DQN(Agent):
    def __init__(self, net, num_actions, config):
        super(DQN, self).__init__()

        if isGPU:
            self.net = net.cuda()
        else:
            self.net = net

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['lr'])
        self.loss = torch.nn.MSELoss()
        self.experience_replay = deque()
        self.action_num = num_actions
        self.batch_size = config['batch_size']
        self.memory_size = config['memory_size']
        self.gamma = config['gamma']
        self.initial_epsilon = config['initial_epsilon']
        self.final_epsilon = config['final_epsilon']
        self.epsilon = self.initial_epsilon
        self.epsilon_decay = config['epsilon_decay']
        self.observation = config['observation']

    def select_action(self, state, test=False):

        on_state = self.preprocess(state, volatile=True)

        greedy = np.random.rand()
        if greedy < self.epsilon and not test:  # explore
            action = np.random.randint(self.action_num)
        else:  # exploit
            if isGPU:
                action = np.argmax(self.net.forward(on_state).data.cpu().numpy())
            else:
                action = np.argmax(self.net.forward(on_state).data.numpy())

        return action

    def update(self, state, action, reward, new_state, done):

        self.experience_replay.append((state, action, reward, new_state, done))  # add new transition to dataset

        if len(self.experience_replay) > self.memory_size:  # if number of examples more than capacity of database, pop
            self.epsilon = max(self.epsilon - (self.initial_epsilon - self.final_epsilon) / self.epsilon_decay, 0)
            self.experience_replay.popleft()

        if len(self.experience_replay) >= self.observation:  # if have enough experience example, go

            minibatch = np.array(random.sample(self.experience_replay, self.batch_size))
            states, actions, rewards, new_states, dones = tuple(minibatch[:, k] for k in range(5))

            new_states = torch.cat([self.preprocess(x) for x in new_states], 0)

            if isGPU:
                q_prime = self.net.forward(new_states).data.cpu().numpy()
            else:
                q_prime = self.net.forward(new_states).data.numpy()

            states = torch.cat([self.preprocess(x) for x in states], 0)
            out = self.net.forward(states)

            # Perform Gradient Descent
            action_input = torch.LongTensor(actions.astype(int))
            y_label = torch.Tensor([rewards[i] if dones[i] else rewards[i] + self.gamma * np.max(q_prime[i])
                                    for i in range(self.batch_size)])

            if isGPU:
                action_input = action_input.cuda()
                y_label = y_label.cuda()

            y_out = out.gather(1, Variable(action_input.unsqueeze(1)))

            self.optimizer.zero_grad()
            loss = self.loss(y_out, Variable(y_label).float())
            loss.backward()
            self.optimizer.step()
