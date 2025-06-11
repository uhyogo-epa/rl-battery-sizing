import numpy as np
import random
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from multiprocessing import Process
import ray
from agent.codesign_drqn import CodesignDRQNagent

# Environment
from rl_env.codesign_energy_env import CodesignEnergyEnv


##########################################################
# Main     
##########################################################
ray.init(ignore_reinit_error = True)
@ray.remote
def run_training(seed, battery_price,mu):
# if __name__ == "__main__":
    random.seed(seed)
    np.random.seed(seed)

    # log_dir = 'logs/test_run_'+datetime.now().strftime('%m%d%H%M')
    # writer  = SummaryWriter(log_dir=log_dir)
    mode= 'discrete'
    # Environment
    env               = CodesignEnergyEnv(mode) 
    state_space       = env.observation_space
    action_space      = env.action_space
    rho_space         = env.rho_space
    #########################################
    # DQN agent
    #########################################
    agent = CodesignDRQNagent(state_space, action_space, rho_space,
                          ## Learning rate
                          learning_rate   = 1e-4,                              
                          ## DQN options 
                          buffer_len=int(100000),
                          lookup_step=24,
                          gamma=0.9,
                          batch_size=8,tau=5e-3,
                          eps_start =0.5,
                          eps_decay=0.995, 
                          eps_end=0.01,
                          random_update= True,
                          min_epi_num=20,
                          target_update_period = 4
                          )
    
    #########################################
    # training
    #########################################
    
    LOG_DIR = f'codesign_drqn_logs_3/cost_{battery_price}_mu_{mu}/test_run_seed_{seed}_cost_{battery_price}_mu_{mu}'
    agent.train(env, 
                EPISODES      = 5000, 
                SHOW_PROGRESS = True, 
                LOG_DIR       = LOG_DIR,
                SAVE_AGENTS   = True, 
                SAVE_FREQ     = 500,
                seed = seed,
                learning_rate_mu    = 1e-5,
                learning_rate_sigma = 0,
                min_epi_codesign    = 300,
                mu    = mu,
                sigma = 0.2,
                battery_price_max=battery_price,
                scheduling_rate = 1,
                scheduling_decay= 0.995
                )
    
if __name__ == "__main__":
    seeds = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  
    battery_price_min = 2000
    battery_price_max = 6000
    solar_radiation_all = np.load("/home/students3/mantani/IEEE_TEMPR/data/sample_data_pv.npy") 
    solar_radiation = solar_radiation_all[4344:4344 + 24*7]
    mus = [max(solar_radiation)*0.5]  

    battery_price = battery_price_min
    while battery_price <= battery_price_max:
        tasks = []
        for mu in mus:
            for seed in seeds:
                task = run_training.remote(seed,battery_price,mu)
                tasks.append(task)
            
        ray.get(tasks)
        
        battery_price += 2000  
# -*- coding: utf-8 -*-

