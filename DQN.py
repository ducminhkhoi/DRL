from globals import *


class DQN(Agent):
    def __init__(self, net, num_actions):
        super(DQN, self).__init__()

        if isGPU:
            self.net = net.cuda()
        else:
            self.net = net

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.loss = torch.nn.MSELoss()
        self.experience_replay = deque()
        self.initial_epsilon = 0.1
        self.final_epsilon = 0.0001
        self.action_num = num_actions
        self.batch_size = 32
        self.memory_size = 50000
        self.gamma = 0.99
        self.epsilon = self.initial_epsilon
        self.observation = 320
        self.explore = 3000000
        self.buffer = []

    def select_action(self, epoch, state):

        on_state = self.preprocess(state, volatile=True)

        greedy = np.random.rand()
        if greedy < self.epsilon or state.size()[0] < 4 or (epoch == 0 and state.size()[0] < 4):  # explore
            action = np.random.randint(self.action_num)
        else:  # exploit
            if isGPU:
                action = np.argmax(self.net.forward(on_state).data.cpu().numpy())
            else:
                action = np.argmax(self.net.forward(on_state).data.numpy())

        if self.epsilon > self.final_epsilon and epoch > self.observation:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        return action

    def update(self, state, action, reward, new_state, done):

        self.experience_replay.append((state, action, reward, new_state, done))  # add new transition to dataset
        # print("Length of experience replay", len(self.experience_replay))

        if len(self.experience_replay) > self.memory_size:  # if number of examples more than capacity of database, pop
            self.experience_replay.popleft()

        if len(self.experience_replay) > self.observation:  # if have enough experience example, go
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
