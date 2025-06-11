import numpy as np
import random
from datetime import datetime
import ray

# Environment
# from rl_env.energy import RenewableEnergyEnv
from rl_env.energy_lstm_codesign import CodesignEnergyLSTMEnv
###############################################################################
from agent.codesign_ddpg  import  CodesignDDPGagent
###############################################################################
ray.init(ignore_reinit_error=True)

@ray.remote
def run_training(seed, battery_price, mu):
# if __name__ == "__main__":
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    # start_time = datetime.now()
    
    # log_dir = 'logs/test_run_'+datetime.now().strftime('%m%d%H%M')
    # writer  = SummaryWriter(log_dir=log_dir)
    
    # Environment
    # rl_env      = RenewableEnergyEnv(mode ='continuous')
    rl_env      = CodesignEnergyLSTMEnv(mode ='continuous')
    
    state_dim   = rl_env.observation_space
    act_dim     = rl_env.action_space
    rho_dim     = rl_env.rho_space
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
    
    # ###############################################
    # # Codesign learning parameters
    # ###############################################
    # learning_rate_mu    = 1e-6
    # learning_rate_sigma = 0
    # min_epi_codesign    = 300
    # rho_list = []
    # reward_list = []
    # mu    = 0.1
    # sigma = 0.2
    # battery_price = 1000

    
    agent = CodesignDDPGagent(state_dim, act_dim, rho_dim, max_act,
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
    LOG_DIR = f'codesign_ddpg_logs_sheduling_decay_0.999_learning_rate_1e-6/battery_{battery_price}_mu_{mu}_10000/test_run_seed_{seed}_battery_{battery_price}_{mu}'
    
    agent.train(rl_env,
              EPISODES      = 5000,
              LOG_DIR       = LOG_DIR,
              SHOW_PROGRESS = True,
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 500,
              RESTART_EP    = None,   
              seed          = seed,
              learning_rate_mu    = 1e-5,
              learning_rate_sigma = 0,
              min_epi_codesign    = 300,
              mu    = mu,
              sigma = 0.2,
              battery_price_max = battery_price,
              scheduling_rate = 1,
              scheduling_decay = 0.999)

if __name__ == "__main__":
    seeds = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  
    battery_price_min = 2000
    battery_price_max = 6000
    solar_radiation_all = np.load("/home/students3/mantani/IEEE_TEMPR/data/sample_data_pv.npy") 
    solar_radiation = solar_radiation_all[4344:4344 + 24*7]
    mus = [max(solar_radiation)*0.5,max(solar_radiation)*1.0]  

    battery_price = battery_price_min
    while battery_price <= battery_price_max:
        tasks = []
        for mu in mus:
            for seed in seeds:
                task = run_training.remote(seed,battery_price,mu)
                tasks.append(task)
            
        ray.get(tasks)
        
        battery_price += 2000  
