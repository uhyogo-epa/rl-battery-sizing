from typing import Dict
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.utils.tensorboard import SummaryWriter

# Q_network
class Q_net(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None, rho_space = None):
        super(Q_net, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."
        assert rho_space is not None, "None rho_space input: rho_space should be selected."
        
        self.hidden_space = 64
        self.state_space = state_space 
        self.action_space = action_space
        self.rho_space = rho_space
        self.state_rho_space = state_space + rho_space
        
        self.Linear1 = nn.Linear(self.state_rho_space, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space,
                            self.hidden_space, batch_first=True)
        self.Linear2 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h, c):
        x = F.relu(self.Linear1(x))
        x, (new_h, new_c) = self.lstm(x, (h, c))
        x = self.Linear2(x)
        return x, new_h, new_c

    def sample_action(self, obs_rho, h, c, epsilon, rho):
        output = self.forward(obs_rho, h, c)

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
                 lookup_step=None,
                 ):
        
        self.random_update = random_update  # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step
        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
            self.memory.append(episode)

    def sample(self, random_sample=False, batch_size=None):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update or random_sample:  # Random upodate       

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
    


def train(q_net=None, target_q_net=None, #episode_memory=None,
          samples=None, 
          seq_len=None, 
          device=None,
          optimizer=None,
          batch_size=8,
          learning_rate=1e-4,
          gamma=0.9,
          rho_space = None):

    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    #samples, seq_len = episode_memory.sample()
    
    # observations = []
    # actions = []
    # rewards = []
    # next_observations = []
    # dones = []
    # rhos = []

    # for i in range(batch_size):
    #     observations.append(samples[i]["obs"])
    #     actions.append(samples[i]["acts"])
    #     rewards.append(samples[i]["rews"])
    #     next_observations.append(samples[i]["next_obs"])
    #     dones.append(samples[i]["done"])
    #     rhos.append(samples[i]["rho"])

    
    # start_time_comparison = time.time()
    observations = [samples[i]["obs"] for i in range(batch_size)]
    actions = [samples[i]["acts"] for i in range(batch_size)]
    rewards = [samples[i]["rews"] for i in range(batch_size)]
    next_observations = [samples[i]["next_obs"] for i in range(batch_size)]
    dones = [samples[i]["done"] for i in range(batch_size)]
    rhos =  [samples[i]["rho"] for i in range(batch_size)]
 
    
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)
    rhos = np.array(rhos) 
    
    observations = torch.FloatTensor(
        observations.reshape(batch_size, seq_len, -1)).to(device)
    actions = torch.LongTensor(actions.reshape(
        batch_size, seq_len, -1)).to(device)
    rewards = torch.FloatTensor(rewards.reshape(
        batch_size, seq_len, -1)).to(device)
    next_observations = torch.FloatTensor(
        next_observations.reshape(batch_size, seq_len, -1)).to(device)
    dones = torch.FloatTensor(dones.reshape(
        batch_size, seq_len, -1)).to(device)
    rhos = torch.FloatTensor(rhos.reshape(batch_size, seq_len, -1)).to(device)
    
    observations_rho = torch.cat([observations, rhos], dim=-1)
    next_observations_rho = torch.cat([next_observations, rhos], dim=-1)
    
    h_target, c_target = target_q_net.init_hidden_state(
        batch_size=batch_size, training=True)

    q_target, _, _ = target_q_net(
        next_observations_rho, h_target.to(device), c_target.to(device))
    
    q_target_max = q_target.max(2)[0].view(batch_size, seq_len, -1).detach()
    targets = rewards + gamma*q_target_max*dones
    
    h, c = q_net.init_hidden_state(batch_size=batch_size, training=True)
    q_out, _, _ = q_net(observations_rho, h.to(device), c.to(device))
    q_a = q_out.gather(2, actions)

    # Multiply Importance Sampling weights to loss
    loss_1 = F.smooth_l1_loss(q_a, targets)

    # Update Network
    optimizer.zero_grad()
    loss_1.backward()
    optimizer.step()
    

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_model(model, path='default.pth'):
    torch.save(model.state_dict(), path)



