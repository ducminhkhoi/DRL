from globals import *
from utils import *
from configs import set_configs
import Agents
from Environments import env_transform

nb_episode = 100000
threshold_reward = 20
is_render = True  # Need to display the game on screen or not
is_new_train = True  # New training or continue training load the old weight from file
is_plotting = True  # Need to plot the rewards over time or not
weight_file = 'weights/{agent_name}_{vis_env_name}_episode_{episode}_reward_{reward}.pkl'
history_length = 4  # Length of consecutive frames to input to Network
in_channels = 4  # Number of channels input to the CNN network, 4 means 4 consecutive frames will be stacked
# to CNN as 4 channels
nb_episode_to_test = 100  # Number of Epochs to test after

"""
Summary inputs: 
 1. input is overlay of 4 consecutive frames
 2. input is 4 consecutive frames stacked as a 4 channel unit.
 
"""

agent_name = 'DQN'  # Change the name of the agent
env_name = 'FlappyBird'  # Change the name of the environment
chosen_config = 'keras'  # or stanford
vis_env_name = env_name+'_'+agent_name+'_'+chosen_config
transform = env_transform[env_name]

config = set_configs[agent_name][chosen_config]

length_to_update = config['length_to_update']  # Start to update the network gradient

env = gym.make(env_name + '-v0')  # Enter name of environment here

agent = getattr(Agents, agent_name)(env.action_space.n, config=config, in_channels=in_channels)

ratios = [1. / (history_length - i) for i in range(history_length)]
ratios = [e / sum(ratios) for e in ratios]

if not is_new_train:
    agent.load(weight_file)

if is_plotting:
    reward_plot = viz.line(np.zeros([1]), env=vis_env_name,
                           opts=dict(title="Rewards over Episode"))
    avg_reward_plot = viz.line(np.zeros([1]), env=vis_env_name,
                               opts=dict(title="Average Reward Over Evaluation"))
    time_reward_plot = viz.line(np.zeros([1]), env=vis_env_name,
                               opts=dict(title="Average Reward Over Time"))


def create_inputs(buf):
    if in_channels == 4:
        result = torch.cat(buf)
    else:
        result = sum([buf[i] * ratios[i] for i in range(len(buf))])
    return result


def run_episodes(e_list_reward, e_reward_plot, is_training=False):
    nb_runs = 1 if is_training else 5
    total_reward = 0

    # Average over number runs
    for step in range(nb_runs):
        state = env.reset()
        buffer = deque()
        buffer.append(transform(state))
        h = 0
        score = 0
        while True:
            if is_render and not is_training:
                viz.image(state.transpose(2, 0, 1), win=image_plot, env=vis_env_name,
                          opts=dict(title="run: {}".format(step), caption="current score: {}".format(score)))
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

            buffer.append(transform(state - background if h > 1 else state))
            if len(buffer) > history_length:
                buffer.popleft()

            # Update information for Agent if has enough history in training mode
            if is_training and h > length_to_update:
                new_inputs = create_inputs(buffer)
                agent.update(inputs, action, reward, new_inputs, done)

            score += reward
            if done:
                break

        total_reward += score

    avg_reward = total_reward * 1.0 / nb_runs
    e_list_reward.append(avg_reward)

    if is_plotting:
        plot_reward(e_reward_plot, e_list_reward, vis_env_name)

    if is_training:
        print("Name {5}, Episode {0}, timesteps {1}, reward {2}, replay {3}, epsilon {4:.2f}"
              .format(i_episode, h, total_reward, len(agent.experience_replay), agent.epsilon, vis_env_name))
    else:
        print('nb of episode: ', nb_runs, 'Evaluation Average Reward:', avg_reward)
        agent.save(weight_file.format(agent_name=agent_name, episode=i_episode, reward=avg_reward,
                                      vis_env_name=vis_env_name))


if __name__ == '__main__':

    list_reward = []
    best_avg_reward = -1000
    list_avg_reward = []

    start = time()
    list_reward_by_time = []
    best_time_reward = -1000

    image_plot = viz.image(np.ones((3, 500, 300)), env=vis_env_name,
                               opts=dict(caption='{}'.format(vis_env_name)))

    for i_episode in range(nb_episode):

        # Training Agent
        run_episodes(list_reward, reward_plot, is_training=True)

        # Test by every hour
        if len(list_reward_by_time) * 3600 < time() - start < len(list_reward_by_time) * 3600 + 60:
            run_episodes(list_reward_by_time, time_reward_plot)

        # Testing Agent every 100 epochs
        if i_episode % nb_episode_to_test == nb_episode_to_test-1:
            run_episodes(list_avg_reward, avg_reward_plot)

    print("Average Total Reward", sum(list_reward) * 1.0 / nb_episode)
    print("Max Total Reward", max(list_reward))
