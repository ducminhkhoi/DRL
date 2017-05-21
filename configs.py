

set_configs = {
    'DQN': {
        'stanford': {
            'length_to_update': 50,
            'lr': 1e-6,
            'batch_size': 32,
            'memory_size': 5000,
            'gamma': 0.9,
            'initial_epsilon': 0.6,
            'final_epsilon': 0.0,
            'epsilon_decay': 1000,
            'observation': 500
        },
        'keras': {
            'length_to_update': 10,
            'lr': 1e-4,
            'batch_size': 32,
            'memory_size': 50000,
            'gamma': 0.99,
            'initial_epsilon': 0.1,
            'final_epsilon': 0.0001,
            'epsilon_decay': 3000000,
            'observation': 3200
        }
    }
}