import numpy as np
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process
import multiprocessing

# Environment
# from rl_env.energy import RenewableEnergyEnv
from rl_env.energy_lstm import EnergyLSTMEnv
###############################################################################
from agent.ddpg  import  DDPGagent
###############################################################################
def run_training(seed, battery_times):
# if __name__ == "__main__":
    # Set seeds
    random.seed(1)
    np.random.seed(1)
    # start_time = datetime.now()
    
    # log_dir = 'logs/test_run_'+datetime.now().strftime('%m%d%H%M')
    # writer  = SummaryWriter(log_dir=log_dir)
    
    # Environment
    # rl_env      = RenewableEnergyEnv(mode ='continuous')
    mode = 'continuous'
    # battery_times = 0.9
    rl_env      = EnergyLSTMEnv(mode,battery_times)
    
    state_dim   = rl_env.observation_space
    act_dim     = rl_env.action_space
    max_act     = 1
    
    #DRQN　parameter
    ACTOR_LEARN_RATE  = 1e-3
    CRITIC_LEARN_RATE = 1e-3
    DISCOUNT          = 0.9
    REPLAY_MEMORY_SIZE= 5000
    REPLAY_MEMORY_MIN = 100
    MINIBATCH_SIZE    = 32
    EXPL_NOISE        = 0.15
    TAU               = 5e-3
    POLICY_NOISE      = 0.5
    NOISE_DECAY       = 0.995
    NOISE_CLIP        = 0.5
    NOISE_MIN         = 0.01
    POLICY_FREQ       = 4
    
    agent = DDPGagent(state_dim, act_dim, max_act,
                 ACTOR_LEARN_RATE,
                 CRITIC_LEARN_RATE,
         		 DISCOUNT,
                 REPLAY_MEMORY_SIZE,
                 REPLAY_MEMORY_MIN,
                 MINIBATCH_SIZE,
                 EXPL_NOISE,      # Std of Gaussian exploration noise
                 TAU       ,
                 POLICY_NOISE,
                 NOISE_DECAY ,
                 NOISE_CLIP  ,
                 NOISE_MIN,
                 POLICY_FREQ )
    
    # LOG_DIR = f'ddpg_logs/test_run_{datetime.now().strftime("%m%d_%H%M")}'
    LOG_DIR = f'ddpg_lstm_logs/battery_{round(battery_times,2)}/test_run_{datetime.now().strftime("%m%d_%H%M")}_seed_{seed}_battery_{round(battery_times,2)}'
    
    agent.train(rl_env,
              EPISODES      = 1000,
              LOG_DIR       = LOG_DIR,
              SHOW_PROGRESS = True,
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 500,
              RESTART_EP    = None,
              seed = seed,
              battery_times = battery_times)

if __name__ == "__main__":
    seeds = [1,2,3]
    processes = []
    battery_times  = 0.8
    battery_times_max = 1.2
    while  battery_times  <=  battery_times_max:
        processes = []
        for seed in seeds:
            p = Process(target=run_training, args=(seed, battery_times))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        battery_times += 0.2