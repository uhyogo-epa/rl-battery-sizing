import numpy as np
import random
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from multiprocessing import Process

from agent.codesign_drqn import CodesignDRQNagent

# Environment
from rl_env.codesign_energy_env import CodesignEnergyEnv


##########################################################
# Main     
##########################################################
def run_training(seed, battery_price,scheduling_rate):
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
    
    LOG_DIR = f'codesign_drqn_logs/cost_{battery_price}_{scheduling_rate:.0e}/test_run_{datetime.now().strftime("%m%d_%H%M")}_seed_{seed}_cost_{battery_price}_{scheduling_rate:.0e}'
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
                mu    = 0.3,
                sigma = 0.2,
                battery_price_max=battery_price,
                scheduling_rate = scheduling_rate
                )
    
if __name__ == "__main__":
    seeds = [1,2]  
    battery_price_min = 4000
    battery_price_max = 4000
    scheduling_rates = [5e-2,1e-2]  # 直接リストで指定

    battery_price = battery_price_min
    while battery_price <= battery_price_max:
        for scheduling_rate in scheduling_rates:  # 指定したリストの値をループ
            processes = []
            for seed in seeds:
                p = Process(target=run_training, args=(seed, battery_price, scheduling_rate))
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()
        
        battery_price += 1000  # 1000ずつ増やす
# -*- coding: utf-8 -*-

