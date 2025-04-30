import numpy as np
import copy
from tqdm import tqdm  # progress bar
import torch.nn.functional as F
import pandas as pd
from   datetime import datetime
import torch
import torch.nn as nn
from   torch.utils.tensorboard import SummaryWriter

device = torch.device("cpu")

# DDPG Agent class
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, action_dim)		
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        action = self.max_action * torch.tanh(self.l3(a))
        return action
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)
    def forward(self, state, action):
        q1 = torch.relu(self.l1(torch.cat([state, action], 1)))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
###########################################################################
# Reply buffer
###########################################################################
class ReplayMemory(object):
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        # buffers
        self.state      = np.zeros((max_size, state_dim))
        self.action     = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward     = np.zeros((max_size, 1))
        self.not_done   = np.zeros((max_size, 1))
        
    def add(self, state, action, next_state, reward, done):
        # buffering
        self.state[self.ptr] = state
        self.action[self.ptr] = action
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
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
            )
    def __len__(self):
        return self.size
    
class DDPGagent:
    def __init__(self, state_dim, action_dim, max_action,
                 ACTOR_LEARN_RATE   = 1e-4,
                 CRITIC_LEARN_RATE  = 1e-4,
         		 DISCOUNT           = 0.9,
                 REPLAY_MEMORY_SIZE = 5000,
                 REPLAY_MEMORY_MIN  = 100,
                 MINIBATCH_SIZE     = 32,
                 EXPL_NOISE        = 0.15,      # Std of Gaussian exploration noise
                 TAU                = 0.005,
                 POLICY_NOISE       = 0.5,
                 NOISE_DECAY        = 1,
                 NOISE_CLIP         = 0.5,
                 NOISE_MIN          = 0.1,
                 POLICY_FREQ        = 4):
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = ACTOR_LEARN_RATE)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = CRITIC_LEARN_RATE)
        self.max_action   = max_action
        
        # Replay Memory
        self.replay_memory = ReplayMemory(state_dim, action_dim, REPLAY_MEMORY_SIZE)
        self.REPLAY_MEMORY_MIN = REPLAY_MEMORY_MIN
        self.MINIBATCH_SIZE    = MINIBATCH_SIZE
        self.action_dim   = action_dim
        self.max_action   = max_action
        self.discount     = DISCOUNT
        self.EXPL_NOISE   = EXPL_NOISE
        self.tau          = TAU
        self.policy_noise = POLICY_NOISE
        self.noise_decay  = NOISE_DECAY
        self.noise_clip   = NOISE_CLIP
        self.noise_min    = NOISE_MIN
        self.policy_freq  = POLICY_FREQ
        self.policy_update_cnt = 0
        
    def get_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def get_value(self, state):
        state  = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        return self.critic(state, action).cpu().data.numpy().flatten()
    
    def train(self, env,
              EPISODES      = 50,
              LOG_DIR       = None,
              SHOW_PROGRESS = True,
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 500,
              RESTART_EP    = None,  
              seed = 1,
              battery_times = 0.1):
        
       ######################################
       # Prepare log writer
       ######################################
       if LOG_DIR:
           summary_dir    = LOG_DIR
           summary_writer = SummaryWriter(log_dir=summary_dir)
       if LOG_DIR and SHOW_PROGRESS:
           print(f'Progress recorded: {summary_dir}')
           print(f'---> $tensorboard --logdir {summary_dir}')
           
       ##########################################
       # Define iterator for training loop
       ##########################################
       start = 0 if RESTART_EP == None else RESTART_EP
       if SHOW_PROGRESS:
           iterator = tqdm(range(start, EPISODES), ascii=True, unit='episodes')
       else:
           iterator = range(start, EPISODES)
           
       bidding_history = []
       current_datetime = datetime.now().strftime('%Y%m%d_%H%M')    
       penalty_history = []
       battery_penalty_history = []
       for episode in iterator:
            # リプレイバッファからランダムにバッチをサンプリング
            ##########################
            # Reset
            ##########################
            state, is_done = env.reset(), False
            episode_reward = 0
            episode_bidding = []
            episode_q0 = self.get_value(state)[0]
            episode_penalty_history = []
            episode_battery_penalty_history = []
            #######################################################
            # Iterate until episode ends
            #######################################################
            while not is_done:
                # get action
                if len(self.replay_memory) < self.REPLAY_MEMORY_MIN:
                    action = self.max_action * ( np.random.rand(self.action_dim) -0.5 ) * 2
                else:
                    noise = np.random.normal(0, self.max_action * self.EXPL_NOISE, size=self.action_dim)
                    action = (self.get_action(state)+noise).clip(-self.max_action, self.max_action)
                # make a step
                next_state, reward, is_done = env.step(action)
                episode_reward += reward
                episode_bidding.append(env.bidding)
                # episode_penalty_history.append(env.penalty)
                # episode_battery_penalty_history.append(env.battery_penalty)
                
                # train Q network
                self.replay_memory.add(state, action, next_state, reward, is_done)
                if episode > 20:
                    self.update_step()
                # update current state
                state = next_state
            bidding_history.append(episode_bidding)   
            penalty_history.append(episode_penalty_history)
            battery_penalty_history.append(episode_battery_penalty_history)
            self.EXPL_NOISE *= self.noise_decay 
            self.EXPL_NOISE  = max(self.EXPL_NOISE,self.noise_min)
            ###################################################
            # Log
            ###################################################
            if LOG_DIR:
                summary_writer.add_scalar("Episode Reward", episode_reward, episode)
                summary_writer.add_scalar("Episode Q0",     episode_q0,     episode)
                summary_writer.add_scalar("noise",     self.EXPL_NOISE,     episode)
               
                summary_writer.flush()
                if SAVE_AGENTS and episode % SAVE_FREQ == 0:
                    ckpt_path = summary_dir + f'/agent-{episode}'
                    torch.save({'actor-weights':  self.actor.state_dict(),
                                'critic-weights': self.critic.state_dict(),
                                }, ckpt_path)
                    
            bidding_df = pd.DataFrame(bidding_history)
            penalty_df = pd.DataFrame(penalty_history)
            battery_penalty_df = pd.DataFrame(battery_penalty_history)
            # Save each DataFrame to CSV files
            bidding_df.to_csv(f'action/episode_bidding_seed_{seed}_{current_datetime}_battery_{round(battery_times,2)}.csv', index=False)
            penalty_df.to_csv(f'action/episode_penalty_seed_{seed}_{current_datetime}_battery_{round(battery_times,2)}.csv', index=False)
            battery_penalty_df.to_csv(f'action/episode_battery_penalty_seed_{seed}_{current_datetime}_battery_{round(battery_times,2)}.csv', index=False)
            
            print(f"Episode {episode + 1}: Reward : {episode_reward}")
            print(f"Episode {episode + 1}: episode_q0: {episode_q0}")
            
            
    ####################################################################
    # Update of actor and critic
    ####################################################################
    def update_step(self):
        ###############################################
        # Calculate Critic loss
        ###############################################
        
        # Sample replay buffer
        state, action, next_state, reward, not_done = self.replay_memory.sample(self.MINIBATCH_SIZE)
        # Calculate target
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
			
            next_action = (
                self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * target_Q
            
        # Compute critic loss
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
       
        #####################################
        # Update neural network weights
        #####################################
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        ###############################################
        # Delayed policy updates
        ###############################################
        self.policy_update_cnt = (self.policy_update_cnt + 1) % self.policy_freq
        if self.policy_update_cnt == 0:
            # Compute actor losse
            actor_loss = -self.critic(state, self.actor(state)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
