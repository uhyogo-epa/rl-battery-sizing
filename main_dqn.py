import numpy as np
import random
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from agent.dqn import DQN
from multiprocessing import Process
import ray
# Environment
# from rl_env.energy import RenewableEnergyEnv
from rl_env.energy_lstm import EnergyLSTMEnv

##########################################################
# Main     
##########################################################
ray.init(ignore_reinit_error = True)

@ray.remote
def run_training(seed,  battery_times, learnig_rate):
# if __name__ == "__main__":
    random.seed(seed)
    np.random.seed(seed)
    # Environment
    # env               = RenewableEnergyEnv(mode = 'discrete')
    mode = 'discrete'
    env               = EnergyLSTMEnv(mode ,battery_times,learning_rate)
    state_space       = env.observation_space
    action_space      = env.action_space

    #########################################
    # DQN agent
    #########################################
    agent = DQN(state_space, action_space,
                          ## Learning rate
                          CRITIC_LEARN_RATE   = 1e-4,  
                          DISCOUNT            = 0.99,                             
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
    
    # LOG_DIR = f'dqn_logs/test_run_{datetime.now().strftime("%m%d_%H%M")}_seed_{seed}'
    LOG_DIR = f'dqn_lstm_logs_com/battery_{round(battery_times,2)}/test_run_seed_{seed}_battery_{round(battery_times,2)}'

    agent.train(env, 
                EPISODES      = 1000, 
                SHOW_PROGRESS = True, 
                LOG_DIR       = LOG_DIR,
                SAVE_AGENTS   = True, 
                SAVE_FREQ     = 500,
                seed = seed,
                battery_times = battery_times
                )


if __name__ == "__main__":
    seeds = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    battery_times =0.4
    battery_times_max = 1.0
    learning_rates = [1e-7]
    while battery_times <= battery_times_max:
        tasks = []
        for learning_rate in learning_rates:
            for seed in seeds:
                task = run_training.remote(seed,battery_times,learning_rate)
                tasks.append(task)
            
        ray.get(tasks)
        
        battery_times += 0.2
