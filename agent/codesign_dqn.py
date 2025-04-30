import copy
from   datetime import datetime
import numpy as np
from   tqdm import tqdm  # progress bar
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.tensorboard import SummaryWriter
import random

device = torch.device("cpu")

########################################################################
# Neural Networks (critic)
########################################################################
class Critic(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None, rho_space = None):
        super(Critic, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.hidden_space = 64
        self.state_space  = state_space
        self.action_space = action_space
        self.rho_space    = rho_space
        
        self.Linear1 = nn.Linear(self.state_space + self.rho_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.Linear3 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, rho):
        x = torch.cat([x, rho], dim=-1)
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        return x
    
    def sample_action(self, x, rho, epsilon):
        x =  torch.FloatTensor(x.reshape(1, -1)).to(device)
        rho = torch.tensor([[rho]], requires_grad=True).to(device)
        output = self.forward(x, rho)
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
        self.rho        = np.zeros((max_size, 1))
        
    def add(self, state, action_idx, next_state, reward, done, rho):
        # buffering
        self.state[self.ptr]      = state
        self.action_idx[self.ptr] = action_idx
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.rho[self.ptr] = rho
        
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
            torch.FloatTensor(self.not_done[ind]).to(device),
            torch.FloatTensor(self.rho[ind]).to(device)
            )
    
    def __len__(self):
        return self.size

###############################################################################
# DQN agent
###############################################################################

class CodesignDQN:
    def __init__(self, state_space, action_space, rho_space, 
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
                 MIN_EPI_NUM         = 20
                 ): 

        # Critic
        self.critic              = Critic(state_space, action_space,rho_space)
        self.critic_target       = copy.deepcopy(self.critic)
        self.critic_optimizer    = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LEARN_RATE)
        self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY

        # Replay Memory
        self.replay_memory     = ReplayMemory(state_space, REPLAY_MEMORY_SIZE)
        self.REPLAY_MEMORY_MIN = REPLAY_MEMORY_MIN
        self.MINIBATCH_SIZE    = MINIBATCH_SIZE
        
        # DQN Options
        self.actNum        = action_space
        self.DISCOUNT      = DISCOUNT
        self.epsilon       = EPSILON_INIT 
        self.EPSILON_INI   = EPSILON_INIT
        self.EPSILON_MIN   = EPSILON_MIN
        self.EPSILON_DECAY = EPSILON_DECAY
        self.min_epi_num   = MIN_EPI_NUM
        # Initialization of variables
        self.target_update_counter = 0
                

    # def get_qs(self, obs, rho):
    #     rho = torch.FloatTensor([[rho]]).to(device).requires_grad_()
    #     obs = torch.FloatTensor(obs.reshape(1, -1)).to(device).requires_grad_()
    #     return self.critic.forward(obs, rho)
    
    
    def get_value(self, obs, rho):
        rho = torch.FloatTensor([[rho]]).to(device).requires_grad_()
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device).requires_grad_()
        q_values = self.critic.forward(obs, rho)
        max_value = q_values.max().item()
        return max_value
    
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
              seed = 1,
              learning_rate_mu    = 1e-6,
              learning_rate_sigma = 0,
              min_epi_codesign    = 300,
              mu    = 0.1,
              sigma = 0.2,
              battery_price=1000
              
              ):
        
        ######################################
        # Prepare log writer 
        ######################################
        if LOG_DIR:
            summary_dir    = LOG_DIR+'/test_run_'+datetime.now().strftime('%m%d_%H%M')
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
                               self.EPSILON_INI*np.power(self.EPSILON_DECAY,RESTART_EP))
        
        rho_list = []
        reward_list = []
 
        #######################################################################
        # Main loop
        #######################################################################
        for episode in iterator:
            
            ##########################
            # Reset
            ##########################
            rho = max(0.05, np.random.normal(mu,sigma))
            state   = env.reset(rho) 
            is_done = False
            episode_reward = 0
            episode_q0     = self.get_value(state, rho)
        
            #######################################################
            # Iterate until episode ends 
            #######################################################
            while not is_done:
                state_rho = np.concatenate([state, np.array([rho])])
                state_rho = torch.from_numpy(state_rho).float().to(device).requires_grad_(True)
                action = self.critic.sample_action(state, rho, self.epsilon)
                
                # make a step
                next_state, reward, is_done = env.step(action,rho)
                episode_reward += reward
                

                # store experience and train Q network
                self.replay_memory.add(state, action, next_state, reward, is_done, rho)
                if episode > self.min_epi_num:
                    self.update_critic()
        
                # update current state
                state = next_state
    
            ################################################
            # Update target Q-function and decay epsilon            
            ################################################
            self.target_update_counter += 1
            if self.target_update_counter > self.UPDATE_TARGET_EVERY:
                self.critic_target.load_state_dict(self.critic.state_dict())
                self.target_update_counter = 0
    
            if self.epsilon > self.EPSILON_MIN:
                self.epsilon *= self.EPSILON_DECAY
                self.epsilon  = max( self.EPSILON_MIN, self.epsilon) 
                

            ###################################################
            # Log
            ###################################################
            if LOG_DIR: 
                summary_writer.add_scalar("Episode Reward", episode_reward, episode)
                summary_writer.add_scalar("Episode Q0",     episode_q0,     episode)
                summary_writer.add_scalar('Mu', mu, episode)

                
                summary_writer.flush()
    
                if SAVE_AGENTS and episode % SAVE_FREQ == 0:
                    
                    ckpt_path = summary_dir + f'/agent-{episode}'
                    torch.save({'critic-weights': self.critic.state_dict(),
                                'target-weights': self.critic_target.state_dict(),
                                }, 
                       ckpt_path)
                
        rho_list.append(rho)
        reward_list.append(reward)        
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
        print(f"Episode {episode + 1}: episode_q0: {episode_q0}")
        print(f"Episode {episode + 1}: Mu:{mu}")
            
    def update_critic(self):

        ###############################################
        # Calculate DQN loss
        ###############################################
        # Sample minibatch
        state, action, next_state, reward, not_done, rho = self.replay_memory.sample(self.MINIBATCH_SIZE)
        
        # Calculate target
        with torch.no_grad():
            next_value = self.critic_target.forward(next_state, rho).max(1, keepdim=True).values
            targetQ    = reward + not_done * self.DISCOUNT * next_value

        # DQN Loss (lossD)
        currentQ = self.critic(state,rho).gather(1, action)        
        loss     = F.mse_loss( currentQ, targetQ)        

        #####################################
        # Update neural network weights 
        #####################################
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return # end: train_step
