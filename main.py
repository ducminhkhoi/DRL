from globals import *
from utils import *
from configs import set_configs
import Agents
import Environments

nb_episode = 100000
threshold_reward = 20
is_render = True  # Need to display the game on screen or not
is_training = True  # Train or Evaluate only
is_new_train = True  # New training or continue training load the old weight from file
is_plotting = True  # Need to plot the rewards over time or not
using_rnn = False  # 1 for not using RNN, 2 for using RNN
name_of_run = str(random.randint(0, 1024))
weight_file = 'weights/{agent_name}_{name_of_run}_episode_{episode}_reward_{reward}.pkl'
history_length = 4  # Length of consecutive frames to input to Network
in_channels = 4  # Number of channels input to the CNN network, 4 means 4 consecutive frames will be stacked
# to CNN as 4 channels

"""
Summary inputs: 
 1. input is overlay of 4 consecutive frames
 2. input is 4 different frames and then put to CNN and RNN
 3. input is 4 consecutive frames stacked as a 4 channel unit.
 
"""

agent_name = 'DQN'  # Change the name of the agent
env_name = 'FlappyBird'  # Change the name of the environment
chosen_config = 'keras'  # or stanford
vis_env_name = env_name+'_'+agent_name+'_'+chosen_config

config = set_configs[agent_name][chosen_config]

length_to_update = config['length_to_update']  # Start to update the network gradient

env = gym.make(env_name + '-v0')  # Enter name of environment here

cnn_net = getattr(Environments, env_name + 'Net')(using_rnn=using_rnn, in_channels=in_channels)
agent = getattr(Agents, agent_name)(cnn_net, env.action_space.n, config=config)

ratios = [1. / (history_length - i) for i in range(history_length)]
ratios = [e / sum(ratios) for e in ratios]

if not is_new_train:
    agent.load(weight_file)

if is_plotting:
    reward_plot = viz.line(np.zeros([1]), env=vis_env_name,
                           opts=dict(title="Rewards over Episode, {} using RNN={}"
                           .format(name_of_run, using_rnn)))
    avg_reward_plot = viz.line(np.zeros([1]), env=vis_env_name,
                               opts=dict(title="Average Reward Over Evaluation, {} using RNN={}".
                                         format(name_of_run, using_rnn)))

if is_render:
    temp_plot = viz.heatmap(np.ones((500, 300)))


def create_inputs(buf):
    if using_rnn or in_channels == 4:
        result = torch.cat(buffer)
    else:
        result = sum([buf[i] * ratios[i] for i in range(len(buf))])
    return result


if __name__ == '__main__':

    list_reward = []
    best_avg_reward = -1000
    list_avg_reward = []

    image_plot = viz.image(np.ones((3, 500, 300)), env=vis_env_name,
                               opts=dict(caption='{} Using RNN={}!'.format(name_of_run, using_rnn)))

    for i_episode in range(nb_episode):
        state = env.reset()

        # Training Agent
        k = 0
        total_reward = 0
        buffer = deque()
        buffer.append(cnn_net.transform(state))

        while True:
            if is_render:
                viz.image(state.transpose(2, 0, 1), win=image_plot, env=vis_env_name)

                if not is_on_server:
                    env.render()

            if k == 1:
                background = state
                background[230:280, 50:100, :] = background[100, 100, :] * np.ones([50, 50, 3])

            if len(buffer) < history_length:  # not enough buffer, just random sample from action_space
                action = env.action_space.sample()
            else:
                # Get action from Agent
                inputs = create_inputs(buffer)
                # viz.heatmap(inputs[0, :, :], win=temp_plot)
                action = agent.select_action(inputs)

            # Receive information from environment
            new_state, reward, done, _ = env.step(action)
            reward = int(reward)

            if reward not in [-1, 0, 1]:
                reward += 4

            buffer.append(cnn_net.transform(new_state - background if k > 1 else new_state))
            if len(buffer) > history_length:
                buffer.popleft()

            # Update information for Agent if has enough history
            if is_training and k > length_to_update:
                new_inputs = create_inputs(buffer)
                agent.update(inputs, action, reward, new_inputs, done)

            # statistic
            total_reward += reward
            k += 1
            state = new_state

            if done:
                print("Name {5}, Episode {0}, timesteps {1}, reward {2}, replay {3}, epsilon {4:.2f}"
                      .format(i_episode, k, total_reward, len(agent.experience_replay), agent.epsilon, name_of_run))
                list_reward.append(total_reward)
                if is_plotting:
                    plot_reward(reward_plot, list_reward, vis_env_name)
                total_reward = 0
                k = 0
                break

        # Testing Agent every 10 epochs
        if i_episode % 100 == 99:
            total_reward = 0

            for step in range(10):
                state = env.reset()
                buffer = deque()
                buffer.append(cnn_net.transform(state))
                h = 0
                while True:
                    viz.image(state.transpose(2, 0, 1), win=image_plot, env=vis_env_name)
                    if not is_on_server:
                        env.render()

                    if h == 1:
                        background = state
                        background[230:280, 50:100, :] = background[100, 100, :] * np.ones([50, 50, 3])

                    if len(buffer) < history_length:  # not enough buffer, just random sample from action_space
                        action = env.action_space.sample()
                    else:
                        # Get action from Agent
                        inputs = create_inputs(buffer)
                        action = agent.select_action(inputs, test=True)

                    state, reward, done, _ = env.step(action)

                    if int(reward) not in [-1, 0, 1]:
                        reward += 4

                    buffer.append(cnn_net.transform(state - background))
                    if len(buffer) > history_length:
                        buffer.popleft()

                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward * 1.0 / 10
            print('episode: ', 10, 'Evaluation Average Reward:', avg_reward)
            list_avg_reward.append(avg_reward)

            if is_plotting:
                plot_reward(avg_reward_plot, list_avg_reward, vis_env_name)

            if avg_reward >= best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(weight_file.format(agent_name=agent_name, episode=i_episode, reward=total_reward,
                                              name_of_run=name_of_run))

    print("Average Total Reward", sum(list_reward) * 1.0 / nb_episode)
    print("Max Total Reward", max(list_reward))
