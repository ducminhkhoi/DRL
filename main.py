from globals import *
from utils import *
from configs import set_configs
import Agents
from Environments import env_transform
from agents.A3C import A3CNet
import sys

nb_episode = 100000
threshold_reward = 20
is_render = True  # Need to display the game on screen or not
is_plotting = True  # Need to plot the rewards over time or not

# to CNN as 4 channels
nb_episode_to_test = 100000  # Number of Epochs to test after

agent_name = 'DQN'  # Change the name of the agent
#env_name = 'FlappyBird'  # Change the name of the environment
env_name = 'MountainCar'
if agent_name == 'DDPG':
    env_name += 'Continuous'
chosen_config = 'config1'  # see in `configs.py` file
version = '-v0'
other_text = 'server_test2' if is_on_server else 'cpu'
vis_env_name = env_name + '_' + agent_name + '_' + chosen_config + ('_' + other_text if other_text else '')
transform = env_transform[env_name]
config = set_configs[agent_name][chosen_config]
history_length = config['history_length']  # Length of consecutive frames to input to Network
in_channels = config[
    'in_channels']  # Number of channels input to the CNN network, 4 means 4 consecutive frames will be stacked
length_to_update = config['length_to_update']  # Start to update the network gradient

viz.close(env=vis_env_name)
prefix = '/scratch/nguyenkh/' if is_on_server else ''


def run_episodes(env, agent, i_episode, image_plot, e_list_reward, e_reward_plot, weight_file, nb_runs=1, is_training=False):
    total_reward = 0

    # Average over number runs
    for step in range(nb_runs):

        state = env.reset()
        if env_name != 'FlappyBird':
            state = env.render(mode='rgb_array')

        buffer = deque(maxlen=history_length)
        buffer.append(transform(state))
        h = 0
        score = 0
        while True:
            if is_render and (not is_training or env_name != 'FlappyBird') and h > 0:
                viz.image(state.transpose(2, 0, 1), win=image_plot, env=vis_env_name,
                          opts=dict(title="run: {}, step {}".format(step, h),
                                    caption="current score: {}, current action: {}".format(score, action)))
                if not is_on_server:
                    env.render()

            if h == 1 and env_name == 'FlappyBird':
                background = state
                background[230:280, 50:100, :] = background[100, 100, :] * np.ones([50, 50, 3])

            if len(buffer) < history_length:  # not enough buffer, just random sample from action_space
                action = env.action_space.sample()
            else:
                # Get action from Agent
                inputs = torch.cat(buffer)
                action = agent.select_action(inputs, test=not is_training)

            if isinstance(action, (list, tuple)):
                action = action[0]

            state, reward, done, _ = env.step(action)
            if env_name != 'FlappyBird':
                state = env.render(mode='rgb_array')
                print(h, action, score)
            else:
                reward = np.clip(reward, -1, 1)

            buffer.append(transform((state - background) if h > 1 and env_name == 'FlappyBird' else state))

            # Update information for Agent if has enough history in training mode
            if is_training and h > length_to_update:
                new_inputs = torch.cat(buffer)
                agent.update(inputs, action, reward, new_inputs, done)

            score += reward
            h += 1
            if done:
                break

        total_reward += score

    avg_reward = total_reward * 1.0 / nb_runs
    e_list_reward.append(avg_reward)

    if is_plotting:
        plot_reward(e_reward_plot, e_list_reward, vis_env_name)

    if is_training:
        print("Name {4}, Episode {0}, timesteps {1}, reward {2}, replay {3}"
              .format(i_episode, h, total_reward, len(agent.experience_replay), vis_env_name))
    else:
        print('nb of episode: ', nb_runs, 'Evaluation Average Reward:', avg_reward)
        agent.save(weight_file.format(episode=i_episode, reward=avg_reward, vis_env_name=vis_env_name))


def a3c_worker_agent(shared_net, rank, queue):
    env = gym.make(env_name + version)

    agent = getattr(Agents, agent_name)(shared_net=shared_net, num_actions=env.action_space.n,
                                        in_channels=in_channels, config=config)
    agent.net.train()

    # if is_render:
    #     image_plot = viz.image(np.ones((3, 500, 300)), env=vis_env_name, opts=dict(caption=''))

    reward_plot = viz.line(np.zeros([1]), env=vis_env_name,
                           opts=dict(title="Rewards over Episode of Rank {}".format(rank)))

    list_reward = []
    i_episode = 0

    print("Worker {} start".format(rank))

    state = env.reset()

    if env_name != 'FlappyBird':
        state = env.render(mode='rgb_array')

    score, h = 0, 0
    buffer = deque(maxlen=history_length)
    buffer.append(transform(state))

    while True:
        values, log_probs, rewards, entropies = [], [], [], []

        # Sync with shared model
        agent.net.load_state_dict(agent.shared_net.state_dict())

        # perform action according to current policy Pi (agent's parameter) to get list of rewards
        for _ in range(config['local_t_max']):

            # if is_render:
            #     if is_on_server:
            #         viz.image(state.transpose(2, 0, 1), win=image_plot, env=vis_env_name,
            #                   opts=dict(title="Rank: {}, run {}".format(rank, i_episode),
            #                             caption="current score: {}".format(score)))
            #     else:
            #         env.render()

            if h == 1 and env_name == 'FlappyBird':
                background = state
                background[230:280, 50:100, :] = background[100, 100, :] * np.ones([50, 50, 3])

            # Select next action base on current state
            if len(buffer) < history_length:  # not enough buffer, just random sample from
                # action_space
                action = env.action_space.sample()

                # Go to next state
                state, reward, done, _ = env.step(action)
            else:
                # Get action from Agent
                inputs = torch.cat(buffer)
                action, value, log_prob, entropy = agent.select_action(inputs)

                # Go to next state
                state, reward, done, _ = env.step(action)

                if env_name != 'FlappyBird':
                    state = env.render(mode='rgb_array')
                else:
                    reward = np.clip(reward, -1, 1)

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)

            score += reward
            h += 1

            count_t_global = queue.get()
            count_t_global += 1
            queue.put(count_t_global)

            # print("Go here worker", rank, count_t_global)

            buffer.append(transform((state - background) if h > 1 and env_name == 'FlappyBird' else state))

            if done:
                print("Name {3}, Rank {4}, Episode {0}, timesteps {1}, reward {2}".format(i_episode, h,
                                                                                          score, vis_env_name, rank))
                list_reward.append(score)
                plot_reward(reward_plot, list_reward, vis_env_name)

                # reset everything
                state = env.reset()
                if env_name != 'FlappyBird':
                    state = env.render(mode='rgb_array')

                score, h = 0, 0
                i_episode += 1
                buffer = deque(maxlen=history_length)
                buffer.append(transform(state))

                break

        agent.update(values, log_probs, rewards, entropies, done, torch.cat(buffer))

        if count_t_global > config['global_t_max']:
            break


def a3c_master_agent(shared_net, queue):
    env = gym.make(env_name + version)

    agent = getattr(Agents, agent_name)(shared_net=shared_net, num_actions=env.action_space.n,
                                        in_channels=in_channels, config=config)

    weight_file = prefix + 'weights/{vis_env_name}_episode_{episode}_reward_{reward}.pt'
    list_avg_reward = []
    list_reward_by_time = []

    image_plot = viz.image(np.ones((3, 500, 300)), env=vis_env_name, opts=dict(caption=''))

    time_reward_plot = viz.line(np.zeros([1]), env=vis_env_name, opts=dict(title="Average Reward Over Time"))

    avg_reward_plot = viz.line(np.zeros([1]), env=vis_env_name,
                               opts=dict(title="Average Reward Over Evaluation"))

    interval_to_test = 1800  # seconds

    start = time.time()

    agent.net.eval()

    print("Master start")

    while True:

        elapsed_time = time.time() - start
        count_t_global = queue.get()
        queue.put(count_t_global)

        # Test by every hour and save to weight file
        if len(list_reward_by_time) * interval_to_test <= elapsed_time <= \
           len(list_reward_by_time) * interval_to_test + 10:
            print("go here")
            # Update parameters from worker
            agent.net.load_state_dict(agent.shared_net.state_dict())

            run_episodes(env, agent, count_t_global, image_plot, list_reward_by_time,
                         time_reward_plot, weight_file, nb_runs=5)

        # Testing Agent every 100 epochs and save to weight file
        if count_t_global % nb_episode_to_test == nb_episode_to_test - 1:
            # Update parameters from worker
            agent.net.load_state_dict(agent.shared_net.state_dict())

            run_episodes(env, agent, count_t_global, image_plot, list_avg_reward,
                         avg_reward_plot, weight_file, nb_runs=5)


if __name__ == '__main__':

    saved_weight_file = None
    if len(sys.argv) > 1:
        saved_weight_file = sys.argv[1]

    list_reward = []
    list_avg_reward = []

    start = time.time()
    list_reward_by_time = []

    reward_plot = viz.line(np.zeros([1]), env=vis_env_name, opts=dict(title="Rewards over Episode"))

    image_plot = viz.image(np.ones((3, 500, 300)), env=vis_env_name, opts=dict(caption=''))

    time_reward_plot = viz.line(np.zeros([1]), env=vis_env_name, opts=dict(title="Average Reward Over Time"))

    avg_reward_plot = viz.line(np.zeros([1]), env=vis_env_name,
                               opts=dict(title="Average Reward Over Evaluation"))

    if agent_name in ['DQN', 'DDPG']:

        weight_file = prefix + 'weights/{vis_env_name}_episode_{episode}_reward_{reward}.pt'

        mode = "train"  # train or test

        if mode == "train":
            if saved_weight_file != None:
            	agent.load(saved_weight_file)
            env = gym.make(env_name + version)  # Enter name of environment here
            agent = getattr(Agents, agent_name)(num_actions=1 if agent_name == 'DDPG' else env.action_space.n,
                                                in_channels=config['in_channels'],
                                                config=config)

            for i_episode in range(nb_episode):

                # Training Agent
                run_episodes(env, agent, i_episode, image_plot, list_reward, reward_plot, weight_file, is_training=True)

                # Test by every hour and save to weight file
                if len(list_reward_by_time) * 3600 < time.time() - start < len(list_reward_by_time) * 3600 + 60:
                    run_episodes(env, agent, i_episode, image_plot, list_reward_by_time,
                                 time_reward_plot, weight_file, nb_runs=5)

                # Testing Agent every 100 epochs and save to weight file
                if i_episode % nb_episode_to_test == nb_episode_to_test - 1:
                    run_episodes(env, agent, i_episode, image_plot, list_avg_reward,
                                 avg_reward_plot, weight_file, nb_runs=5)

        else:
            print("testing the learning model")
            weight_file = 'weights/DQN_370_episode_7099_reward_1222.0.pt'
            env = gym.make(env_name + version)  # Enter name of environment here
            agent = getattr(Agents, agent_name)(num_actions=env.action_space.n, in_channels=in_channels,
                                                config=config)

            agent.load(weight_file)
            run_episodes(env, agent, 0, image_plot, list_reward, reward_plot, "", nb_runs=100)

    elif agent_name == 'A3C':

        mode = 'test'  # train or test

        if mode == 'train':
            processes = []
            num_processes = config['parallel_agent_size']
            count_t_global = 0
            queue = mp.Queue()
            queue.put(count_t_global)

            shared_net = A3CNet(config, in_channels, config['num_actions'])
            if isGPU:
                shared_net = shared_net.cuda()
            shared_net.share_memory()

            # add master agent to monitor
            p = mp.Process(target=a3c_master_agent, args=[shared_net, queue])
            p.start()
            processes.append(p)

            # add worker agent to learn
            for rank in range(num_processes):
                p = mp.Process(target=a3c_worker_agent, args=(shared_net, rank, queue))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

        else:
            print("testing the learning model")
            weight_file = 'weights/FlappyBird_A3C_FF_server_test2_episode_13309999_reward_182.6.pt'
            env = gym.make(env_name + version)  # Enter name of environment here

            shared_net = A3CNet(config, in_channels, config['num_actions'])
            if isGPU:
                shared_net = shared_net.cuda()
            shared_net.share_memory()
            agent = getattr(Agents, agent_name)(num_actions=env.action_space.n, in_channels=in_channels,
                                                config=config, shared_net=shared_net)

            agent.load(weight_file)
            run_episodes(env, agent, 0, image_plot, list_reward, reward_plot, "", nb_runs=100)
