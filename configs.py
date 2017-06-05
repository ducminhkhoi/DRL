

set_configs = {
    'DQN': {
        'config1': {  # stanford
            'length_to_update': 50,
            'lr': 1e-6,
            'batch_size': 32,
            'memory_size': 5000,
            'gamma': 0.9,
            'initial_epsilon': 0.6,
            'final_epsilon': 0.0,
            'epsilon_decay': 1000,
            'observation': 32,
            'history_length': 4,
            'in_channels': 4,
            'num_actions': 2
        },
        'config2': {  # keras
            'length_to_update': 10,
            'lr': 1e-4,
            'batch_size': 32,
            'memory_size': 50000,
            'gamma': 0.99,
            'initial_epsilon': 0.1,
            'final_epsilon': 0.0001,
            'epsilon_decay': 3000000,
            'observation': 3200,
            'history_length': 4,
            'in_channels': 4,
            'num_actions': 2
        }
    },
    'A3C': {
        'config1': {  # LSTM
            'name': 'LSTM',
            'length_to_update': 5,
            'lr': 1e-4,
            'gamma': 0.99,
            'tau': 1,
            'hidden_size': 512,
            'clip_norm': 40,
            'entropy_beta': 0.01,
            'parallel_agent_size': 4,
            'history_length': 4,
            'in_channels': 4,
            'local_t_max': 32,
            'global_t_max': 40000000,
            'num_actions': 2
        },
        'config2': {
            'name': 'FF',
            'length_to_update': 5,
            'lr': 1e-4,
            'gamma': 0.99,
            'tau': 1,
            'hidden_size': 512,
            'clip_norm': 40,
            'entropy_beta': 0.01,
            'parallel_agent_size': 4,
            'history_length': 4,
            'in_channels': 4,
            'local_t_max': 32,
            'global_t_max': 40000000,
            'num_actions': 2
        }
    },
    'DDPG':{
        'config1': {  # official setting
            'length_to_update': 5,
            'name': 'config1',
            'lr': 0.001,   # learning rate
            'plr': 0.0001,  # policy learning rate
            'hidden1': 10,
            'hidden2': 200,
            'init_w': 0.003,
            'ou_theta': 0.15,
            'ou_sigma': 0.2,
            'ou_mu': 0.0,
            'tau': 0.001,
            'batch_size': 32,
            'discount': 0.99,
            'epsilon_decay': 10000,
            'min_epsilon': 0.1,
            'max_epsilon': 1.,
            'memory_size': 1000000,
            'observation': 10000,  # change to 10000
            'history_length': 4,
            'in_channels': 4,
            'num_actions': 1,
            'use_memory': False,
            'action_type': 'continuous',
            'use_expect': False
        },
        'config2': {
            'length_to_update': 50,
            'name': 'config2',
            'lr': 1e-6,   # learning rate
            'plr': 1e-6,  # policy learning rate
            'hidden1': 20,
            'hidden2': 300,
            'hidden3': 600,
            'hidden4': 600,
            'init_w': 0.003,
            'ou_theta': 0.15,
            'ou_sigma': 0.2,
            'ou_mu': 0.0,
            'tau': 0.001,
            'batch_size': 32,
            'discount': 0.9,
            'epsilon_decay': 10000,
            'memory_size': 5000,
            'observation': 500,
            'history_length': 4,
            'in_channels': 4,
            'num_actions': 2,
            'use_memory': False,
            'action_type': 'continuous',
        }

    }

}