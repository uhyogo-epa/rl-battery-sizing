import numpy as np
import random
import torch
import multiprocessing
import os
from datetime import datetime
import ray

# Environment
from rl_env.energy import RenewableEnergyEnv

###############################################################################
# Main func with pytorch
from agent.drqn import DRQNagent

ray.init(ignore_reinit_error = True)

@ray.remote
def run_training(seed, battery_times):
# if __name__ == "__main__":
    random.seed(seed)
    np.random.seed(seed)
    
    ###########################################    
    # Environment setting
    ##########################################
    env = RenewableEnergyEnv(mode= 'discrete',battery_times=battery_times)
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
    
    LOG_DIR = f'drqn_logs_1/battery_{round(battery_times,2)}/test_run_seed_{seed}_battery_{round(battery_times,2)}'
    os.makedirs(LOG_DIR, exist_ok=True) 
    
    agent.train(env,
          EPISODES      = 1000,
          LOG_DIR       = LOG_DIR,
          SHOW_PROGRESS = True,
          SAVE_AGENTS   = True,
          SAVE_FREQ     = 500,
          #RESTART_EP    = None,
          seed          = seed,
          battery_times = battery_times
          )
    
if __name__ == "__main__":
    seeds = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    battery_times = 0.1
    battery_times_max = 1.4
    while battery_times <= battery_times_max:
        tasks = []
        for seed in seeds:
            task = run_training.remote(seed,battery_times)
            tasks.append(task)
        
        ray.get(tasks)
        
        battery_times += 0.1
