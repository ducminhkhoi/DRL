
from globals import *
from utils import *
from configs import set_configs
import Agents
from Environments import env_transform

nb_episode = 100000
threshold_reward = 20
is_render = True  # Need to display the game on screen or not
is_plotting = True  # Need to plot the rewards over time or not

# to CNN as 4 channels
nb_episode_to_test = 100  # Number of Epochs to test after

"""
Summary inputs: 
 1. input is overlay of 4 consecutive frames
 2. input is 4 consecutive frames stacked as a 4 channel unit.

"""

agent_name = 'DDPG'  # Change the name of the agent
env_name = 'FlappyBird'  # Change the name of the environment
chosen_config = 'config1'  # see in `configs.py` file
other_text = 'server_test2' if is_on_server else 'cpu'
vis_env_name = env_name + '_' + agent_name + '_' + chosen_config + ('_' + other_text if other_text else '')
transform = env_transform[env_name][agent_name]
config = set_configs[agent_name][chosen_config]
history_length = config['history_length']  # Length of consecutive frames to input to Network
in_channels = config[
    'in_channels']  # Number of channels input to the CNN network, 4 means 4 consecutive frames will be stacked
length_to_update = config['length_to_update']  # Start to update the network gradient

viz.close(env=vis_env_name)
prefix = '/scratch/nguyenkh/' if is_on_server else ''


def run_episodes(env, agent, e_list_reward, e_reward_plot, nb_runs=1, is_training=False):
    total_reward = 0

    # Average over number runs
    for step in range(nb_runs):
        state = env.reset()
        buffer = deque(maxlen=history_length)
        buffer.append(transform(state))
        h = 0
        score = 0
        while True:
            if is_render and not is_training:
                if is_on_server:
                    viz.image(state.transpose(2, 0, 1), win=image_plot, env=vis_env_name,
                              opts=dict(title="run: {}".format(step), caption="current score: {}".format(score)))
                else:
                    env.render()

            if h == 1:
                background = state
                background[230:280, 50:100, :] = background[100, 100, :] * np.ones([50, 50, 3])

            if len(buffer) < history_length:  # not enough buffer, just random sample from action_space
                dis_action = env.action_space.sample()
            else:
                # Get action from Agent
                inputs = torch.cat(buffer)
                action = agent.select_action(inputs, test=not is_training)
                dis_action = action
                if isinstance(action, np.ndarray):
                    # dis_action = np.clip(dis_action, -1, 1)
                    # dis_action = softmax(dis_action) if config['num_actions'] > 1 else sigmoid(dis_action)
                    dis_action = np.abs(dis_action)
                    dis_action = np.argmax(dis_action)

                print(dis_action, action)

            state, reward, done, _ = env.step(dis_action)

            reward = np.clip(reward, -1, 1)

            buffer.append(transform((state - background) if h > 1 else state))

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
        print("Name {5}, Episode {0}, timesteps {1}, reward {2}, replay {3}, epsilon {4:.2f}"
              .format(i_episode, h, total_reward, len(agent.experience_replay), agent.epsilon, vis_env_name))
    else:
        print('nb of episode: ', nb_runs, 'Evaluation Average Reward:', avg_reward)
        if mode == 'train':
            agent.save(weight_file.format(episode=i_episode, reward=avg_reward, vis_env_name=vis_env_name))


def a3c_worker_agent(env, agent, rank, count_t_global):
    if is_render:
        image_plot = viz.image(np.ones((3, 500, 300)), env=vis_env_name, opts=dict(caption=''))

    reward_plot = viz.line(np.zeros([1]), env=vis_env_name,
                           opts=dict(title="Rewards over Episode of Rank {}".format(rank)))

    list_reward = []
    i_episode = 0

    state = env.reset()
    score, h = 0, 0
    buffer = deque(maxlen=history_length)
    buffer.append(transform(state))

    while True:
        values, log_probs, rewards, entropies = [], [], [], []

        # perform action according to current policy Pi (agent's parameter) to get list of rewards
        for _ in range(config['local_t_max']):

            if is_render:
                if is_on_server:
                    viz.image(state.transpose(2, 0, 1), win=image_plot, env=vis_env_name,
                              opts=dict(title="Rank: {}, run {}".format(rank, i_episode),
                                        caption="current score: {}".format(score)))
                else:
                    env.render()

            if h == 1:
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
                action, value, log_prob, entropy = agent.select_action(torch.cat(buffer))

                # Go to next state
                state, reward, done, _ = env.step(action)
                reward = np.clip(reward, -1, 1)

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)

            score += reward
            h += 1
            count_t_global.value += 1

            buffer.append(transform((state - background) if h > 1 else state))

            if done:
                print("Name {3}, Rank {4}, Episode {0}, timesteps {1}, reward {2}".format(i_episode, h,
                                                                                          score, vis_env_name, rank))
                list_reward.append(score)
                plot_reward(reward_plot, list_reward, vis_env_name)

                # reset everything
                state = env.reset()
                score, h = 0, 0
                i_episode += 1
                buffer = deque()
                buffer.append(transform(state))

                break

        agent.update(values, log_probs, rewards, entropies, done, torch.cat(buffer))

        if count_t_global.value > config['global_t_max']:
            break


def a3c_master_agent(env, agent):
    nb_runs = 5
    interval_to_test = 1800  # seconds
    weight_file = prefix + 'weights/{vis_env_name}_time_{time}_reward_{reward}.pt'

    start = time.time()

    test_image_plot = viz.image(np.ones((3, 500, 300)), env=vis_env_name, opts=dict(caption=''))

    time_reward_plot = viz.line(np.zeros([1]), env=vis_env_name, opts=dict(title="Average Reward Over Time"))

    list_reward_by_time = []

    while True:
        elapsed_time = time.time() - start

        if len(list_reward_by_time) * interval_to_test <= elapsed_time <= \
                                len(list_reward_by_time) * interval_to_test + 5:

            # Update parameters from worker
            agent.net.load_state_dict(agent.shared_net.state_dict())

            total_score = 0
            for step in range(nb_runs):
                score, h = 0, 0
                state = env.reset()
                buffer = deque(maxlen=history_length)
                buffer.append(transform(state))
                while True:

                    if is_on_server:
                        viz.image(state.transpose(2, 0, 1), win=test_image_plot, env=vis_env_name,
                                  opts=dict(title="Master Test, run {}".format(step),
                                            caption="current score: {}".format(score)))
                    else:
                        env.render()

                    if h == 1:
                        background = state
                        background[230:280, 50:100, :] = background[100, 100, :] * np.ones([50, 50, 3])

                    # Select next action base on current state
                    if len(buffer) < history_length:  # not enough buffer, just random sample from action_space
                        action = env.action_space.sample()

                        # Go to next state
                        state, reward, done, _ = env.step(action)
                    else:
                        # Get action from Agent
                        inputs = torch.cat(buffer)
                        action, value, log_prob, entropy = agent.select_action(inputs)

                        # Go to next state
                        state, reward, done, _ = env.step(action)
                        reward = np.clip(reward, -1, 1)

                    score += reward
                    h += 1

                    buffer.append(transform((state - background) if h > 1 else state))

                    if done:
                        break

                total_score += score

            avg_score = total_score * 1. / nb_runs
            list_reward_by_time.append(avg_score)
            plot_reward(time_reward_plot, list_reward_by_time, vis_env_name)
            print('nb of episode: ', nb_runs, 'Evaluation Average Reward:', avg_score)
            if avg_score > 2.:
                agent.save(weight_file.format(vis_env_name=vis_env_name, time=int(elapsed_time),
                                              reward=avg_score))


if __name__ == '__main__':

    if agent_name in ['DQN', 'DDPG']:
        weight_file = prefix + 'weights/{vis_env_name}_episode_{episode}_reward_{reward}.pt'
        list_reward = []
        list_avg_reward = []

        start = time.time()
        list_reward_by_time = []

        reward_plot = viz.line(np.zeros([1]), env=vis_env_name, opts=dict(title="Rewards over Episode"))

        image_plot = viz.image(np.ones((3, 500, 300)), env=vis_env_name, opts=dict(caption=''))

        time_reward_plot = viz.line(np.zeros([1]), env=vis_env_name, opts=dict(title="Average Reward Over Time"))

        avg_reward_plot = viz.line(np.zeros([1]), env=vis_env_name,
                                   opts=dict(title="Average Reward Over Evaluation"))

        mode = "train"  # train or test

        if mode == "train":
            env = gym.make(env_name + '-v0')  # Enter name of environment here
            agent = getattr(Agents, agent_name)(num_actions=config['num_actions'],
                                                in_channels=config['in_channels'],
                                                config=config)

            for i_episode in range(nb_episode):

                # Training Agent
                run_episodes(env, agent, list_reward, reward_plot, is_training=True)

                # Test by every hour and save to weight file
                if len(list_reward_by_time) * 3600 < time.time() - start < len(list_reward_by_time) * 3600 + 60:
                    run_episodes(env, agent, list_reward_by_time, time_reward_plot, nb_runs=5)

                # Testing Agent every 100 epochs and save to weight file
                if i_episode % nb_episode_to_test == nb_episode_to_test - 1:
                    run_episodes(env, agent, list_avg_reward, avg_reward_plot, nb_runs=5)

        else:
            print("testing the learning model")
            weight_file = 'weights/DQN_370_episode_7099_reward_1222.0.pt'
            env = gym.make(env_name + '-v0')  # Enter name of environment here
            agent = getattr(Agents, agent_name)(num_actions=env.action_space.n, in_channels=in_channels,
                                                config=config)

            agent.load(weight_file)
            run_episodes(env, agent, list_reward, reward_plot, nb_runs=100)

    elif agent_name == 'A3C':

        mode = 'train'  # train or test

        if mode == 'train':
            processes = []
            num_processes = config['parallel_agent_size']
            count_t_global = Value('i', 0)

            # create list of envs and agents
            envs = [gym.make(env_name + '-v0') for _ in range(num_processes + 1)]
            for i, env in enumerate(envs):
                env.seed(i)

            from agents.A3C import A3CNet

            shared_net = A3CNet(config, in_channels, envs[0].action_space.n)
            shared_net.share_memory()

            agents = [getattr(Agents, agent_name)(shared_net=shared_net, num_actions=envs[0].action_space.n,
                                                  in_channels=in_channels, config=config) for _ in
                      range(num_processes + 1)]

            # add master agent to monitor
            p = mp.Process(target=a3c_master_agent, args=(envs[0], agents[0]))
            p.start()
            processes.append(p)

            # add worker agent to learn
            for rank in range(1, num_processes + 1):
                p = mp.Process(target=a3c_worker_agent, args=(envs[rank], agents[rank], rank, count_t_global))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

        else:
            print("testing the learning model")

            pass

            # print("Average Total Reward", sum(list_reward) * 1.0 / nb_episode)
            # print("Max Total Reward", max(list_reward))
