import numpy as np
import random
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from agent.dqn import DQN
from multiprocessing import Process

# Environment
# from rl_env.energy import RenewableEnergyEnv
from rl_env.energy_lstm import EnergyLSTMEnv

##########################################################
# Main     
##########################################################
# def run_training(seed,  battery_times):
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    # Environment
    # env               = RenewableEnergyEnv(mode = 'discrete')
    mode = 'discrete'
    battery_times = 1.0
    env               = EnergyLSTMEnv(mode ,battery_times)
    state_space       = env.observation_space
    action_space      = env.action_space

    #########################################
    # DQN agent
    #########################################
    agent = DQN(state_space, action_space,
                          ## Learning rate
                          CRITIC_LEARN_RATE   = 1e-4,  
                          DISCOUNT            = 0.9,                             
                          ## DQN options 
                          REPLAY_MEMORY_SIZE  = 5000, 
                          REPLAY_MEMORY_MIN   = 100,
                          MINIBATCH_SIZE      = 8,                              
                          UPDATE_TARGET_EVERY = 4,
                          EPSILON_INIT        = 0.5,
                          EPSILON_DECAY       = 0.995, 
                          EPSILON_MIN         = 0.01, 
                          )
    
    #########################################
    # training
    #########################################
    
    LOG_DIR = f'dqn_logs/test_run_{datetime.now().strftime("%m%d_%H%M")}'
    # LOG_DIR = f'dqn_lstm_logs/test_run_{datetime.now().strftime("%m%d_%H%M")}_seed_{seed}_battery_{round(battery_times,2)}'

    agent.train(env, 
                EPISODES      = 1000, 
                SHOW_PROGRESS = True, 
                LOG_DIR       = LOG_DIR,
                SAVE_AGENTS   = True, 
                SAVE_FREQ     = 500,
                seed = 1,
                )

# if __name__ == "__main__":
#     seeds = [1,2,3,4,5]  
#     processes = []
#     battery_times  = 0.25
#     battery_times_max = 1.0
#     while  battery_times  <=  battery_times_max:
#         processes = []
#         for seed in seeds:
#             p = Process(target=run_training, args=(seed, battery_times ))
#             p.start()
#             processes.append(p)
        
#         for p in processes:
#             p.join()
        
#         battery_times += 0.25