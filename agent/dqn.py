import os
import copy
from   datetime import datetime
import numpy as np
import pandas as pd
from   tqdm import tqdm  # progress bar
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.tensorboard import SummaryWriter
import random
import time
import csv

device = torch.device("cpu")

########################################################################
# Neural Networks (critic)
########################################################################
class Critic(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None):
        super(Critic, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.hidden_space = 64
        self.state_space  = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.Linear3 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = self.Linear3(x)
        return x
    
    def sample_action(self, obs, epsilon):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)  
        output = self.forward(obs)

        if random.random() < epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            return output.argmax().item()

###########################################################################
# Reply buffer
###########################################################################

class ReplayMemory(object):
    def __init__(self, state_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        # buffers
        self.state      = np.zeros((max_size, state_dim))
        self.action_idx = np.zeros((max_size, 1), dtype=np.int32)
        self.next_state = np.zeros((max_size, state_dim))
        self.reward     = np.zeros((max_size, 1))
        self.not_done   = np.zeros((max_size, 1))

    def add(self, state, action_idx, next_state, reward, done):
        # buffering
        self.state[self.ptr]      = state
        self.action_idx[self.ptr] = action_idx
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        # move pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.LongTensor(self.action_idx[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
            )
    
    def __len__(self):
        return self.size

###############################################################################
# DQN agent
###############################################################################

class DQN:
    def __init__(self, state_dim, action_num, 
                 ## DQN options
                 CRITIC_LEARN_RATE   = 1e-3,
                 DISCOUNT            = 0.99, 
                 REPLAY_MEMORY_SIZE  = 5_000,
                 REPLAY_MEMORY_MIN   = 100,
                 MINIBATCH_SIZE      = 16, 
                 UPDATE_TARGET_EVERY = 20, 
                 EPSILON_INIT        = 1,
                 EPSILON_DECAY       = 0.998, 
                 EPSILON_MIN         = 0.01,
                 ): 

        # Critic
        self.critic              = Critic(state_dim, action_num)
        self.critic_target       = copy.deepcopy(self.critic)
        self.critic_optimizer    = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LEARN_RATE)
        self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY

        # Replay Memory
        self.replay_memory     = ReplayMemory(state_dim, REPLAY_MEMORY_SIZE)
        self.REPLAY_MEMORY_MIN = REPLAY_MEMORY_MIN
        self.MINIBATCH_SIZE    = MINIBATCH_SIZE
        
        # DQN Options
        self.actNum        = action_num
        self.DISCOUNT      = DISCOUNT
        self.epsilon       = EPSILON_INIT 
        self.EPSILON_INIT   = EPSILON_INIT
        self.EPSILON_MIN   = EPSILON_MIN
        self.EPSILON_DECAY = EPSILON_DECAY
        
        # Initialization of variables
        self.target_update_counter = 0
                

    def get_qs(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)  
        return self.critic.forward(state)
    

    ####################################################################
    # Training loop
    ####################################################################
    def train(self, env, 
              EPISODES      = 50, 
              LOG_DIR       = None,
              SHOW_PROGRESS = True,
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 1,
              RESTART_EP    = None,
              seed =1,
              battery_times =1
              ):
        
        
        ######################################
        # Prepare log writer 
        ######################################
        if LOG_DIR:
            summary_dir    = LOG_DIR
            summary_writer = SummaryWriter(log_dir=summary_dir)      
    
        ##########################################
        # Define iterator for training loop
        ##########################################
        start = 0 if RESTART_EP == None else RESTART_EP
        if SHOW_PROGRESS:    
            iterator = tqdm(range(start, EPISODES), ascii=True, unit='episodes')        
        else:
            iterator = range(start, EPISODES)
    
        if RESTART_EP:
            self.epsilon = max(self.EPSILON_MIN, 
                               self.EPSILON_INIT*np.power(self.EPSILON_DECAY,RESTART_EP))
        bidding_history = []
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M')
        penalty_history = []
        battery_penalty_history = []
        
        #######################################################################
        # Main loop
        #######################################################################
        for episode in iterator:
            
            ##########################
            # Reset
            ##########################
            state   = env.reset() 
            is_done = False
            episode_bidding = []
            episode_reward = 0
            episode_q0     = self.get_qs(state)
            max_q_values   = episode_q0.max().item()
            episode_penalty_history = []
            episode_battery_penalty_history = []
            #######################################################
            # Iterate until episode ends 
            #######################################################
            while not is_done:
                action = self.critic.sample_action(state, self.epsilon)
                
                # make a step
                next_state, reward, is_done = env.step(action)
                episode_reward += reward
                episode_bidding.append(env.bidding)
                episode_penalty_history.append(env.penalty)
                episode_battery_penalty_history.append(env.battery_penalty)
                
                # store experience and train Q network
                self.replay_memory.add(state, action, next_state, reward, is_done)
                if episode > 20:
                    self.update_critic()
        
                # update current state
                state = next_state
            bidding_history.append(episode_bidding)
            penalty_history.append(episode_penalty_history)
            battery_penalty_history.append(episode_battery_penalty_history)
                
    
            ################################################
            # Update target Q-function and decay epsilon            
            ################################################
            self.target_update_counter += 1
            if self.target_update_counter > self.UPDATE_TARGET_EVERY:
                self.critic_target.load_state_dict(self.critic.state_dict())
                self.target_update_counter = 0
                
            if self.epsilon > self.EPSILON_MIN:
               self.epsilon  = max( self.EPSILON_MIN, self.epsilon* self.EPSILON_DECAY)        
        
            ###################################################
            # Log
            ###################################################
            if LOG_DIR: 
                summary_writer.add_scalar("Episode Reward", episode_reward, episode)
                summary_writer.add_scalar("Episode Q0",     max_q_values,     episode)
                summary_writer.add_scalar("epsilon",     self.epsilon,     episode)
                
                summary_writer.flush()
    
                if SAVE_AGENTS and episode % SAVE_FREQ == 0:
                    
                    ckpt_path = summary_dir + f'/agent-{episode}'
                    torch.save({'critic-weights': self.critic.state_dict(),
                                'target-weights': self.critic_target.state_dict(),
                                }, 
                       ckpt_path)
                    
            print(f"Episode {episode + 1}: Reward : {episode_reward}")
            print(f"Episode {episode + 1}: episode_q0: {max_q_values}")
        
            bidding_df = pd.DataFrame(bidding_history)
            penalty_df = pd.DataFrame(penalty_history)
            battery_penalty_df = pd.DataFrame(battery_penalty_history)
            
            # Save each DataFrame to CSV files
            bidding_df.to_csv(f'dqn_action/episode_bidding_seed_{seed}_battery_{round(battery_times,2)}_{current_datetime}.csv', index=False)
            penalty_df.to_csv(f'dqn_action/episode_penalty_seed_{seed}_battery_{round(battery_times,2)}_{current_datetime}.csv', index=False)
            battery_penalty_df.to_csv(f'dqn_action/episode_battery_penalty_seed_{seed}_battery_{round(battery_times,2)}_{current_datetime}.csv', index=False)
            
   
    def update_critic(self):

        ###############################################
        # Calculate DQN loss
        ###############################################
        # Sample minibatch
        state, action, next_state, reward, not_done = self.replay_memory.sample(self.MINIBATCH_SIZE)

        # Calculate target
        with torch.no_grad():
            next_value = self.critic_target.forward(next_state).max(1, keepdim=True)[0]
            targetQ    = reward + not_done * self.DISCOUNT * next_value

        # DQN Loss (lossD)
        currentQ = self.critic(state).gather(1, action)        
        loss     = F.mse_loss( currentQ, targetQ )        
        #####################################
        # Update neural network weights 
        #####################################
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        # print(f"Comprehension version duration: {calcurate_duration:.8f} seconds")
        
        # with open(log_file, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([batch_duration, calcurate_duration])
        return # end: train_step
    
    
        # print(f"Durations saved to {log_file}")

    
