config = {
     'msac':
    {
        'env_name': 'GazeboEnv_test2',
        # 'buffer_size': 1e6, # 50000
        'buffer_size': 50000,
        'actor_learn_freq': 2,
        'update_iteration': 10,
        'target_update_freq': 5,
        'actor_lr': 3e-4, # 3e-3
        'critic_lr': 3e-3,
        # 'lr': 1e-4, # notice 3e-3
        'batch_size': 1024, # 1024
        'hidden_dim': 300,
        'episodes': 1000,
        'max_step': 200,
        'SAVE_DIR': '/save/sac_',
        'PKL_DIR': '/pkl/sac_',
        'LOG_DIR': '/logs', 
        'POLT_NAME': 'SAC_',
    },
    'td3':
    {
        'env_name': 'GazeboEnv',
        'buffer_size': 50000,
        'actor_learn_freq': 2,
        'update_iteration': 10,
        'target_update_freq': 5,
        'actor_lr': 3e-4,
        'critic_lr': 3e-3,
        'batch_size': 1024,
        'hidden_dim': 300,
        'episodes': 1000,
        'max_step': 200,
        'SAVE_DIR': '/save/td3_',
        'PKL_DIR': '/pkl/td3_',
        'LOG_DIR': '/logs', 
        'POLT_NAME': 'TD3_',
    },
    'ddpg': # copy from test_ddpg
    {
        'env_name': 'Pendulum-v0',
        'buffer_size': 50000,
        'actor_learn_freq': 1,
        'update_iteration': 10,
        'target_update_freq': 5,
        'actor_lr': 3e-4,
        'critic_lr': 3e-3,
        'batch_size': 1024,
        'hidden_dim': 300,
        'episodes': 1000,
        'max_step': 200,
        'SAVE_DIR': '/save/ddpg_',
        'PKL_DIR': '/pkl/ddpg_',
        'LOG_DIR': '/logs', 
        'POLT_NAME': 'DDPG_',
    },
    # 'sac':
    # {
    #     'env_name': 'Pendulum-v0',
    #     'buffer_size': 50000,
    #     'actor_learn_freq': 2,
    #     'update_iteration': 10,
    #     'target_update_freq': 10,
    #     'lr': 3e-3,
    #     'batch_size': 128,
    #     'hidden_dim': 32,
    #     'episodes': 2000,
    #     'max_step': 300,
    #     'SAVE_DIR': '/save/sac_',
    #     'PKL_DIR': '/pkl/sac_',
    #     'LOG_DIR': '/logs', 
    #     'POLT_NAME': 'SAC_',
    # },
}