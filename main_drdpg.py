import numpy as np
import random
import torch
import os
from   datetime import datetime
from multiprocessing import Process
import ray
# Environment
from rl_env.energy import RenewableEnergyEnv

###############################################################################
# Main func with pytorch
from agent.drdpg import DRDPGagent
ray.init(ignore_reinit_error = True)

@ray.remote
def run_training(seed, battery_times):
# if __name__ == "__main__":
    random.seed(seed)
    np.random.seed(seed)
    #LOG_DIR =  'drdpg_logs/'
    
    
    ###########################################    
    # Environment setting
    ##########################################
    env = RenewableEnergyEnv(mode= 'continuous',battery_times=battery_times)
    state_space       = env.observation_space
    action_space      = env.action_space
    # rho_space       = 1
    max_action        = 1
    
    ###########################################
    # DRQN setting
    ###########################################
    buffer_len           = int(100000)
    batch_size           = 8
    Actor_learning_rate  = 1e-3
    Critic_learning_rate = 1e-3
    max_epi_num          = 100
    min_epi_num          = 20
    
    episodes             = 1000 
    lookup_step          = 24
    gamma                = 0.99
    tau                  = 5e-3
    
    #noise parameter
    initial_noise           = 1.0
    noise_decay             = 0.996
    noise_min               = 0.15
    
    # DRDPG param
    random_update = True # If you want to do random update instead of sequential update
    lookup_step = 24 * 1# If you want to do random update instead of sequential update
    max_epi_len = 700
    
    # Create Q functions and optimizer
    device = torch.device("cpu")
    
    agent = DRDPGagent(state_space, action_space, max_action, buffer_len, lookup_step, Actor_learning_rate, Critic_learning_rate,
                       gamma, batch_size,tau ,initial_noise,noise_decay,noise_min, random_update,min_epi_num)
    
    LOG_DIR = f'drdpg_logs_com/battery_{round(battery_times,2)}/test_run_seed_{seed}_battery_{round(battery_times,2)}'
    # LOG_DIR = f'drdpg_logss/test_run_{datetime.now().strftime("%m%d_%H%M")}_seed_{seed}'
    
    os.makedirs(LOG_DIR, exist_ok=True) 
    
    agent.train(env,
              EPISODES      = 1000,
              LOG_DIR       = LOG_DIR,
              SHOW_PROGRESS = True, 
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 500,
              RESTART_EP    = None,
              seed = seed
              )
    
    
if __name__ == "__main__":
    seeds = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    battery_times =0.4
    battery_times_max = 1.2
    while battery_times <= battery_times_max:
        tasks = []
        for seed in seeds:
            task = run_training.remote(seed,battery_times)
            tasks.append(task)
        
        ray.get(tasks)
        
        battery_times += 0.2
  
