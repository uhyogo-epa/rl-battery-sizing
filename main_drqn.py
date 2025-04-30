import numpy as np
import random
import torch
import multiprocessing
import os
from datetime import datetime
from multiprocessing import Process

# Environment
from rl_env.energy import RenewableEnergyEnv

###############################################################################
# Main func with pytorch
from agent.drqn import DRQNagent

# def run_training(seed, battery_times):
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    
    ###########################################    
    # Environment setting
    ##########################################
    env = RenewableEnergyEnv(mode='discrete',battery_times=0.5)
    observation_space = env.observation_space
    action_space      = env.action_space
    
    ############################################
    # DRQN setting parameters
    ############################################
    batch_size = 8
    learning_rate = 1e-4
    buffer_len = int(100000)
    min_epi_num = 20 # Start moment to train the Q network
    target_update_period = 4
    eps_start = 0.5
    eps_end = 0.01
    eps_decay = 0.995
    tau = 5e-3
    gamma     = 0.9
    
    # DRQN param
    random_update = True # If you want to do random update instead of sequential update
    lookup_step = 24 * 1# If you want to do random update instead of sequential update
    ##################################################
    # Create Q functions and optimizer

    agent = DRQNagent(observation_space,action_space,buffer_len,lookup_step,learning_rate,gamma,
                      batch_size,tau,eps_start,eps_decay, eps_end,random_update,min_epi_num,target_update_period)
    
    # LOG_DIR = f'drqn_logs/battery_{round(battery_times,2)}/test_run_{datetime.now().strftime("%m%d_%H%M")}_seed_{seed}_battery_{round(battery_times,2)}'
    LOG_DIR = f'drqn_logs/test_run_{datetime.now().strftime("%m%d_%H%M")}'
    os.makedirs(LOG_DIR, exist_ok=True) 
    
    agent.train(env,
          EPISODES      = 1000,
          LOG_DIR       = LOG_DIR,
          SHOW_PROGRESS = True,
          SAVE_AGENTS   = True,
          SAVE_FREQ     = 1,
          RESTART_EP    = None ,
          # seed          = seed,
          # battery_times = battery_times
          )
    
# if __name__ == "__main__":
#     num_cores = multiprocessing.cpu_count()
#     seeds = list(range(num_cores))
#     processes = []
#     battery_times = 0.1
#     battery_times_max = 1.3
#     while battery_times <= battery_times_max:
#         processes = []
#         for seed in seeds:
#             p = Process(target=run_training, args=(seed,battery_times))
#             p.start()
#             processes.append(p)
        
#         for p in processes:
#             p.join()
        
#         battery_times += 0.1