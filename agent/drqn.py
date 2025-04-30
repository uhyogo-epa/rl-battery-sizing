from typing import Dict
import collections
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from   datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cpu")

# Q_network
class Q_net(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None):
        super(Q_net, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.hidden_space = 32
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space,
                            self.hidden_space, batch_first=True)
        self.Linear2 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h, c):
        x = F.relu(self.Linear1(x))
        x, (new_h, new_c) = self.lstm(x, (h, c))
        x = self.Linear2(x)
        return x, new_h, new_c

    def sample_action(self, obs, h, c, epsilon):
        output = self.forward(obs, h, c)

        if random.random() < epsilon:
            return random.randint(0, 1), output[1], output[2]
        else:
            return output[0].argmax().item(), output[1], output[2]

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])


class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False,
                 max_epi_num=100, max_epi_len=700,
                 batch_size=1,
                 lookup_step=None):
        
        self.random_update = random_update  # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step
        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update:  # Random upodate
            sampled_episodes = random.sample(self.memory, self.batch_size)

            check_flag = True  # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                # get minimum step from sampled episodes
                min_step = min(min_step, len(episode))

            for episode in sampled_episodes:
                if min_step > self.lookup_step:  # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                    sample = episode.sample(
                        random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    # sample buffer with minstep size
                    idx = np.random.randint(0, len(episode)-min_step+1)
                    sample = episode.sample(
                        random_update=self.random_update, lookup_step=min_step, idx=idx)
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


class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)

class DRQNagent:
    def __init__(self, observation_space,action_space,buffer_len,lookup_step,learning_rate,gamma,
                      batch_size,tau,eps_start,eps_decay, eps_end,random_update,min_epi_num,target_update_period): 
        
        
        #critic setting
        self.critic = Q_net(observation_space,action_space).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = learning_rate)

        #DRQN parameter
        self.gamma = gamma
        self.batch_size = batch_size      
        self.min_epi_num  = min_epi_num
        
        self.random_update = True
        self.lookup_step = lookup_step
        
        # Replay Memory
        self.memory = EpisodeBuffer()
        self.max_epi_num = 100
        self.max_epi_len = 1000
        self.episode_memory = EpisodeMemory(self.random_update,self.max_epi_num, self.max_epi_len,self.batch_size,lookup_step)
        
        #epsilon parameter
        self.eps_start   = eps_start
        self.eps_end  = eps_end
        self.eps_decay = eps_decay
        
        #policy parameter
        self.tau = tau
        self.target_update_period = target_update_period


    def train(self, env, 
              EPISODES      = 50,
             LOG_DIR       = None,
             SHOW_PROGRESS = True,
             SAVE_AGENTS   = True,
             SAVE_FREQ     = 1,
             RESTART_EP    = None
              ):
            
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
        epsilon = self.eps_start
        
        # Train
        for episode in iterator:
            obs = env.reset()
            done = False
            h, c = self.critic.init_hidden_state(self.batch_size, training=False)
            episode_bidding = []  # List to collect bidding data

            episode_record = EpisodeBuffer()

            episode_reward = 0
            episode_reward_discount = 0
            self.policy_update_cnt = 0
            while not done:

                # Get action
                a, h, c = self.critic.sample_action(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), 
                                                  h.to(device), c.to(device),
                                                  epsilon)

                # Do action
                s_prime, r, done = env.step(a)
                obs_prime = s_prime
                episode_bidding.append(env.bidding)  # Collect bidding data
                
                with torch.no_grad():
                    episode_q0, _, _ = self.critic(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), h, c)
                    max_q_value = episode_q0.max().item()
                    
                done_mask = 0.0 if done else 1.0

                episode_record.put([obs, a, r, obs_prime, done_mask])

                obs = obs_prime
                
                episode_reward += r
                
                episode_reward_discount = r + self.gamma*episode_reward_discount
                
                if episode >= self.min_epi_num:
                    self.update_step()

                if done:
                    break
                
            epsilon = max(self.eps_end, epsilon * self.eps_decay) # Linear annealing
            
            self.episode_memory.put(episode_record)

            bidding_history.append(episode_bidding)  # Save episode bidding data
            
            print(f"Episode {episode+ 1}: Reward : {episode_reward}")
            print(f"Episode {episode + 1}: Reward_discount : {episode_reward_discount}")
            
            ###################################################
            # Log
            ###################################################
            if LOG_DIR:
                summary_writer.add_scalar("Episode Reward", episode_reward, episode)
                summary_writer.add_scalar("Episode Q0",     max_q_value,     episode)
                summary_writer.add_scalar("epsilon",     epsilon,     episode)
               
                summary_writer.flush()
        summary_writer.close()
            
        
        # Save step rewards to a CSV file

        # torch.save(Q.state_dict(),f'Q_net/Q_net_{current_datetime}.pth')
        # torch.save(Q_target.state_dict(),f'Q_net/Q_target_net_{current_datetime}.pth')
        
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
        assert device is not None, "None Device input: device should be selected."
        
        samples, seq_len = self.episode_memory.sample()
    
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
    
        observations = [samples[i]["obs"] for i in range(self.batch_size)]
        actions = [samples[i]["acts"] for i in range(self.batch_size)]
        rewards = [samples[i]["rews"] for i in range(self.batch_size)]
        next_observations = [samples[i]["next_obs"] for i in range(self.batch_size)]
        dones = [samples[i]["done"] for i in range(self.batch_size)]
        
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)
    
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
        
        # end_time_batch = time.time()
        # batch_duration = end_time_batch - start_time_batch
        # print(f"Comprehension version duration: {batch_duration:.8f} seconds")
        
        # start_time_calucurate = time.time()
    
        # Target 
        h_target, c_target = self.critic_target.init_hidden_state(
                            self.batch_size, training=True)
        
        with torch.no_grad():
            q_target, _, _ = self.critic_target(next_observations, 
                                          h_target.to(device), c_target.to(device))
            q_target_max = q_target.max(2)[0].view(self.batch_size, seq_len, -1) #.detach()
        targets      = rewards + self.gamma*q_target_max*dones
    
        # 
        h, c = self.critic.init_hidden_state(self.batch_size, training=True)
        q_out, _, _ = self.critic(observations, h.to(device), c.to(device))
        q_a = q_out.gather(2, actions)
    
        # Multiply Importance Sampling weights to loss
        loss = F.smooth_l1_loss(q_a, targets)
    
        # Update Network
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        # end_time_calcurate = time.time()
        # calcurate_duration = end_time_calcurate - start_time_calucurate
        
        # print(f"Comprehension version duration: {calcurate_duration:.8f} seconds")
        
        # with open(log_file, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([batch_duration, calcurate_duration])
        #     # print(f"Durations saved to {log_file}")
        
        self.policy_update_cnt = (self.policy_update_cnt + 1) % self.target_update_period
        if self.policy_update_cnt == 0:
            # Q_target.load_state_dict(Q.state_dict()) <- navie update
            for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()): # <- soft update
                    target_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)
        
    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    
    
    def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)
