import numpy as np
import copy
import collections
import random
import torch.nn.functional as F
import torch
from   datetime import datetime
import pandas as pd
from tqdm import tqdm  # progress bar
import torch.nn as nn
from typing import Dict
from   torch.utils.tensorboard import SummaryWriter

device = torch.device("cpu")

# DRDPG Agent class
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, rho_dim, max_action):
        super(Actor, self).__init__()
        self.hidden_space = 32
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rho_dim = rho_dim
        self.max_action = max_action
    
        self.Linear1 = nn.Linear(self.state_dim+rho_dim, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first = True)
        self.Linear2 = nn.Linear(self.hidden_space, self.action_dim)
    
    def forward(self, state, rho, h, c):
        x = torch.cat([state, rho],dim=2)
        x = F.relu(self.Linear1(x))
        x, (new_h, new_c) = self.lstm(x, (h, c)) 
        action = self.max_action * torch.tanh(self.Linear2(x))
        return action, new_h, new_c
   
    def init_hidden_state(self, batch_size, training=None):
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])

#DRDPG Critic class
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,rho_dim):
        super(Critic, self).__init__(), 
        self.hidden_space = 32
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rho_dim    = rho_dim
        self.Linear1 = nn.Linear(self.state_dim + self.action_dim + self.rho_dim, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first = True)
        self.Linear2 = nn.Linear(self.hidden_space, 1)
    
    def forward(self, state, action, rho,h, c):
        x = torch.cat([state, action,rho],dim=2)
        x = F.relu(self.Linear1(x))
        # LSTM
        x, (new_h, new_c) = self.lstm(x, (h, c))
        q_value = self.Linear2(x)
        return q_value, new_h, new_c

    def init_hidden_state(self, batch_size, training=None):
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])

###########################################################################
# Reply buffer
###########################################################################

class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []
        self.rho = []
        
    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])
        self.rho.append(transition[5])  
        
    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)
        rho = np.array(self.rho)
        
        if random_update is True:
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]
            rho = rho[idx:idx+lookup_step] 
            
        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done,
                    rho = rho)

    def __len__(self) -> int:
        return len(self.obs)
    
class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False,
                  max_epi_num=100, max_epi_len=700,
                  batch_size=1,
                  lookup_step=None):
        
        self.random_update = random_update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)
        
    def sample(self,random_sample=False, batch_size=None):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update or random_sample:  # Random update       

            if batch_size:        
                sampled_episodes = random.sample(self.memory, batch_size)
            else:
                sampled_episodes = random.sample(self.memory, self.batch_size)

            check_flag = True  # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                # get minimum step from sampled episodes
                min_step = min(min_step, len(episode))

            for episode in sampled_episodes:
                if min_step > self.lookup_step:  # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                    sample = episode.sample(random_update=True, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    # sample buffer with minstep size
                    idx = np.random.randint(0, len(episode)-min_step+1)
                    sample = episode.sample(random_update=True, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)
                    
        ##################### SEQUENTIAL UPDATE ############################
        else:  # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(
                random_update=self.random_update))

        # buffers, sequence_length
        return sampled_buffer, len(sampled_buffer[0]['obs'])

                
    def __len__(self):
        return len(self.memory)
        
class CodesignDRDPGagent:
    def __init__(self, state_dim, action_dim, rho_dim,max_action, 
                 buffer_len, lookup_step, Actor_learning_rate, Critic_learning_rate,
                 gamma, batch_size,tau ,initial_noise, noise_decay, noise_min, random_update, min_epi_num): 
        
        #actor setting
        self.actor = Actor(state_dim, action_dim, rho_dim,max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = Actor_learning_rate)
        
        #critic setting
        self.critic = Critic(state_dim, action_dim,rho_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = Critic_learning_rate)
        
        
        #DRQN parameter
        self.action_dim  = action_dim
        self.max_action  = max_action
        self.gamma = gamma
        self.batch_size = batch_size
        self.lookup_step = lookup_step
        self.min_epi_num  = min_epi_num
        self.max_epi_num = 100
        self.max_epi_len = 1000
        self.random_update = True
        
        # Replay Memory
        self.memory = EpisodeBuffer()
        self.episode_memory = EpisodeMemory(self.random_update,self.max_epi_num, self.max_epi_len,self.batch_size,lookup_step)
        
        #noise parameter
        self.noise_cnt_min = 24*7*10
        self.noise_cnt = 0
        self.noise_init   = initial_noise
        self.noise_decay  = noise_decay
        self.noise_min    = noise_min
        
        #policy parameter
        self.policy_freq = 4
        self.tau          = tau
        self.policy_update_cnt = 0
    
    def get_action(self, state, rho, h, c, noise):
        action, new_h, new_c = self.actor(state,rho, h, c)      
        action = action.cpu().data.numpy().flatten()
        self.noise_cnt += 1
        if self.noise_cnt < self.noise_cnt_min:
            action = self.max_action * (np.random.rand(self.action_dim)-0.5)*2
        else:
            noise_value = np.random.normal(0, self.max_action * noise, size=self.action_dim)
            action = (action + noise_value).clip(-self.max_action, self.max_action)
        return action, new_h, new_c
    
    def get_value(self, state, action, rho,h, c):
        q_value,new_h, new_c = self.critic(state, action, rho,h, c)
        q_value= q_value.cpu().data.numpy().flatten()
        return q_value, new_h, new_c
    
    def train(self, env,
              EPISODES      = 50,
              LOG_DIR       = None,
              SHOW_PROGRESS = True,
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 1,
              RESTART_EP    = None,  
              seed = 1):
        
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
        
        bidding_history = []  # Initialize bidding_history
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M')
        noise = self.noise_init
        ###############################################
        # Codesign learning parameters
        ###############################################
        learning_rate_mu    = 1e-6
        learning_rate_sigma = 0
        min_epi_codesign    = 300
        rho_list = []
        reward_list = []
        mu    = 0.4
        sigma = 0.2
        phi   = [mu, sigma]
        
        # Train
        for episode in iterator:
            rho = max(0.05, np.random.normal(mu,sigma))
            battery_price = 3000
            obs = env.reset(rho)
            done = False

            episode_reward = 0
            episode_reward_discount = 0
            episode_bidding = []  # List to collect bidding data
            episode_record = EpisodeBuffer()
            
            h_actor, c_actor = self.actor.init_hidden_state(self.batch_size, training=False)
            
            h_critic, c_critic = self.critic.init_hidden_state(self.batch_size, training=False)
            
            while not done:

                # Get action
                a, h_actor, c_actor = self.get_action(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), torch.tensor(rho).unsqueeze(0).unsqueeze(0).unsqueeze(0), 
                                                  h_actor.to(device), c_actor.to(device),
                                                  noise)

                # Do action
                obs_prime, r, done = env.step(a,rho)
                
                episode_bidding.append(env.bidding)  # Collect bidding data

                # episode_penalty_history.append(env.penalty)
                # episode_battery_penalty_history.append(env.battery_penalty)
                # episode_xD_t_history.append(env.xD_t)
                
                with torch.no_grad():
                    q_values, h_critic, c_critic = self.get_value(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), torch.from_numpy(a).float().to(device).unsqueeze(0).unsqueeze(0),torch.tensor(rho).unsqueeze(0).unsqueeze(0).unsqueeze(0), h_critic, c_critic)
                    
                    
                    # if t == 0:
                    #     initial_q_value = max_q_value
                # make data
                done_mask = 0.0 if done else 1.0

                episode_record.put([obs, a, r, obs_prime, done_mask,rho])

                obs = obs_prime
                
                episode_reward += r
                
                episode_reward_discount = r + self.gamma*episode_reward_discount

                        
                if episode >= self.min_epi_num:
                    self.update_step()
                # update current state
                    
            noise *= self.noise_decay 
            noise  = max(noise, self.noise_min)
            bidding_history.append(episode_bidding)  # Save episode bidding data
            rho_list.append(rho)
            reward_list.append(episode_reward)
            self.episode_memory.put(episode_record)
            if episode >= min_epi_codesign:
                 if episode % 10 ==0:
            
                    rhos    = np.array(rho_list[-10:])
                    
                    G = np.array(reward_list[-10:])*365/7 - rhos * battery_price
                    # G       = q_max - rhos * battery_price  # Profit - battery_capacity * battery_price
                    mu_grad =  (((rhos - mu) / (sigma ** 2)) * (G -G.mean())).mean()
                    mu      = mu     + learning_rate_mu * mu_grad
        
                    # update sigma            
                    sigma_grad = 0
                    sigma = sigma + learning_rate_sigma * sigma_grad
                    #  (rho_tensor - mu) ** 2  - sigma ** 2) / (sigma ** 3)
            
            print(f"Episode {episode + 1}: Reward : {episode_reward}")
            print(f"Episode {episode + 1}: Reward_discount : {episode_reward_discount}")
            print(f"Episode {episode + 1}: Mu:{mu}")
            
            ###################################################
            # Log
            ###################################################
            if LOG_DIR:
                summary_writer.add_scalar("Episode Reward", episode_reward, episode)
                summary_writer.add_scalar("Episode Q0",     q_values,     episode)
                summary_writer.add_scalar("noise",     noise,     episode)
                summary_writer.add_scalar('Mu', mu, episode)
                summary_writer.flush()
        summary_writer.close()
                
        bidding_df = pd.DataFrame(bidding_history)
        
        # Save each DataFrame to CSV files
        bidding_df.to_csv(f'action/episode_bidding_{current_datetime}.csv', index=False)
        
    ####################################################################
    # Update of actor and critic
    ####################################################################
    def update_step(self):
        ###############################################
        # Calculate Critic loss
        ###############################################
         # Get batch from replay buffer
        samples, seq_len = self.episode_memory.sample()
    
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        rhos =[]
        
        observations = [samples[i]["obs"] for i in range(self.batch_size)]
        actions = [samples[i]["acts"] for i in range(self.batch_size)]
        rewards = [samples[i]["rews"] for i in range(self.batch_size)]
        next_observations = [samples[i]["next_obs"] for i in range(self.batch_size)]
        dones = [samples[i]["done"] for i in range(self.batch_size)]
        rhos =  [samples[i]["rho"] for i in range(self.batch_size)]
        # print(f'observations:{observations_1}')
        
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)
        rhos = np.array(rhos) 
        
        observations = torch.FloatTensor(
            observations.reshape(self.batch_size, seq_len, -1)).to(device)
        actions = torch.LongTensor(actions.reshape(
            self.batch_size, seq_len, -1)).to(device)
        rewards = torch.FloatTensor(rewards.reshape(
            self.batch_size, seq_len, -1)).to(device)
        next_observations = torch.FloatTensor(
            next_observations.reshape(self.batch_size, seq_len, -1)).to(device)
        dones = torch.FloatTensor(dones.reshape(
            self.batch_size, seq_len, -1)).to(device)
        rhos = torch.FloatTensor(rhos.reshape(self.batch_size, seq_len, -1)).to(device)
        
        #######################################################################
        # ターゲットの計算
        ######################################
        # アクターのLSTMの初期状態を取得
        h_actor_target, c_actor_target = self.actor_target.init_hidden_state(self.batch_size, training=True)
        h_actor_target, c_actor_target = h_actor_target.detach(), c_actor_target.detach()

        # クリティックのLSTMの初期状態を取得
        h_critic_target, c_critic_target = self.critic_target.init_hidden_state(self.batch_size, training=True)
        h_critic_target, c_critic_target = h_critic_target.detach(), c_critic_target.detach()

        next_actions, _, _ = self.actor_target(next_observations, rhos, h_actor_target, c_actor_target)
        
        # ターゲットクリティックネットワークでQ値を計算
        target_q_values, _, _ = self.critic_target(next_observations, next_actions, rhos, h_critic_target, c_critic_target)
    
        # ターゲット値を計算
        target_q_values = rewards + self.gamma * target_q_values * dones
        
        #######################################################################
        # アクターとクリティックの更新
        ######################################
        # クリティックネットワークの計算
        h_critic, c_critic = self.critic.init_hidden_state(self.batch_size, training=True)
        
        current_q_values, _, _ = self.critic(observations, actions, rhos, h_critic, c_critic)
        
        # クリティックの損失
        critic_loss = F.mse_loss(current_q_values, target_q_values)
    
        # クリティックの更新
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # アクターの損失計算
        self.policy_update_cnt = (self.policy_update_cnt + 1) % self.policy_freq
        if self.policy_update_cnt == 0:
            h_actor, c_actor = self.actor.init_hidden_state(self.batch_size, training=True)
            predicted_actions, h_actor_next, c_actor_next = self.actor(observations, rhos, h_actor, c_actor)
            actor_loss = -self.critic(observations, predicted_actions, rhos,h_critic, c_critic)[0].mean()
        
            # アクターの更新
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        




    

