
from globals import *
import Agents
import Environments

nb_episode = 100000
threshold_reward = 20
is_render = True  # Need to display the game on screen or not
is_training = True  # Train or Evaluate only
is_new_train = True  # New training or continue training load the old weight from file
is_plotting = True  # Need to plot the rewards over time or not
weight_file = 'weights/{agent_name}_episode_{episode}_reward_{reward}.pkl'
history_length = 4

agent_name = 'DQN'  # Change the name of the agent
env_name = 'FlappyBird'  # Change the name of the environment

env = gym.make(env_name+'-v0')  # Enter name of environment here

cnn_net = getattr(Environments, env_name+'Net')(history_length)
agent = getattr(Agents, agent_name)(cnn_net, env.action_space.n)

if not is_new_train:
    agent.load(weight_file)

reward_plot = viz.line(np.zeros([1]), opts=dict(title="Rewards over Episode"))
avg_reward_plot = viz.line(np.zeros([1]), opts=dict(title="Average Reward Over Evaluation"))

if __name__ == '__main__':

    list_reward = []
    best_avg_reward = -1000
    list_avg_reward = []

    if is_on_server:
        image_plot = viz.image(np.ones((3, 500, 300)))

    for i_episode in range(nb_episode):
        state = env.reset()

        # Training Agent
        k = 0
        total_reward = 0
        buffer = deque()
        buffer.append(state)

        while True:
            if is_render:
                if is_on_server:
                    viz.image(state.transpose(2, 0, 1), win=image_plot)
                else:
                    env.render()

            if k < history_length:  # not enough buffer, just random sample from action_space
                action = env.action_space.sample()
            else:
                # Get action from Agent
                inputs = torch.cat([cnn_net.transform(buffer[i]) for i in range(len(buffer))], 0)
                action = agent.select_action(k + 1, inputs)

            # Receive information from environment
            new_state, reward, done, _ = env.step(action)

            if int(reward) not in [-1, 0, 1]:
                reward += 5

            buffer.append(new_state)
            if len(buffer) > history_length:
                buffer.popleft()

            # Update information for Agent if has enough history
            if is_training and k > 50:
                new_inputs = torch.cat([cnn_net.transform(buffer[i]) for i in range(len(buffer))], 0)
                agent.update(inputs, action, reward, new_inputs, done)

            # statistic
            total_reward += reward
            k += 1
            state = new_state

            if done:
                print("Episode {} finished after {} timesteps with total reward {}, length of experience replay {} " \
                      .format(i_episode, k, total_reward, len(agent.experience_replay)))
                list_reward.append(total_reward)
                if is_plotting:
                    plot_reward(reward_plot, list_reward)
                total_reward = 0
                k = 0
                break

        # Testing Agent every 10 epochs
        if i_episode % 10 == 9:
            total_reward = 0
            buffer = deque()
            buffer.append(state)
            for step in range(10):
                state = env.reset()
                while True:
                    if is_on_server:
                        viz.image(state.transpose(2, 0, 1), win=image_plot)
                    else:
                        env.render()

                    if len(buffer) < history_length:  # not enough buffer, just random sample from action_space
                        action = env.action_space.sample()
                    else:
                        # Get action from Agent
                        inputs = torch.cat(
                            [cnn_net.transform(buffer[i]) for i in range(1, len(buffer))], 0)
                        action = agent.select_action(0, inputs)

                    state, reward, done, _ = env.step(action)

                    if int(reward) not in [-1, 0, 1]:
                        reward += 5

                    buffer.append(state)
                    if len(buffer) > history_length:
                        buffer.popleft()

                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward * 1.0 / 10
            print('episode: ', 10, 'Evaluation Average Reward:', avg_reward)
            list_avg_reward.append(avg_reward)

            if is_plotting:
                plot_reward(avg_reward_plot, list_avg_reward)

            if avg_reward >= best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(weight_file.format(agent_name=agent_name, episode=i_episode, reward=total_reward))

    print("Average Total Reward", sum(list_reward)*1.0/nb_episode)
    print("Max Total Reward", max(list_reward))
