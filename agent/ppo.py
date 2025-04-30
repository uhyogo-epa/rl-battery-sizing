import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from   datetime import datetime
from   torch.utils.tensorboard import SummaryWriter
# from   tqdm import tqdm

# PPO Actor-Critic Classes
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False):
        super(ActorCritic, self).__init__()
        self.continuous = continuous
        hidden_dim      = 64
        
        self.actor_Linear1 = nn.Linear(state_dim, hidden_dim)
        self.actor_Linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor layers
        if continuous:
            self.mean_layer = nn.Linear(hidden_dim, action_dim) 
            self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        else:
           self.action_layer = nn.Linear(hidden_dim, action_dim)
           
        # Critic layer
        self.critic_Linear1 = nn.Linear(state_dim, hidden_dim)
        self.critic_Linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)


    def forward(self, state):
        x = torch.relu(self.actor_Linear1(state))
        x = torch.relu(self.actor_Linear2(x))
        if self.continuous:
            mean = self.mean_layer(x)
            log_std = self.log_std_layer(x).clamp(-20, 2)  # 数値の安定性を確保
            std = torch.exp(log_std)
            return mean, std
        else:
            action_probs = torch.softmax(self.action_layer(x), dim=-1)
            return action_probs
        
    def get_value(self, state):
        # Critic forward pass
        v = torch.relu(self.critic_Linear1(state))
        v = torch.relu(self.critic_Linear2(v))
        return self.value_layer(v)

    def get_action(self, state):
        if self.continuous:
            mean, std = self.forward(state)
            dist = Normal(mean, std)
            action = dist.sample()
            return action.detach().numpy(), dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1)
        
        else:
            action_probs = self.forward(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action), dist.entropy()
        
    def evaluate_action(self, state, action):
        if self.continuous:
            mean, std = self.forward(state)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)  
            entropy = dist.entropy().sum(dim=-1)         
        else:
            action_probs = self.forward(state)
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(action)  
            entropy = dist.entropy()          

        value = self.get_value(state).squeeze(-1)  # 状態価値
        
        return log_prob, entropy, value

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate,  continuous=False, gamma=0.99, clip_eps=0.2,ent_coef= 0.1, gae_lambda=0.95):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.continuous = continuous
        self.learning_rate = learning_rate
        self.lambda_t   = gae_lambda
        self.actor_critic = ActorCritic(state_dim, action_dim, continuous)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
          
    def compute_gae(self, rewards, values,  next_value, dones):
        delta = rewards + self.gamma * next_value * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = delta[t] + self.gamma * self.lambda_t * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return returns, advantages
    
    def ppo_update(self, states, actions, log_probs, returns, advantages):
        advantages = advantages.detach()
        log_probs = log_probs.detach()
        # 新しい log_prob, エントロピー, 値関数を計算
        new_log_probs, entropy, values = self.actor_critic.evaluate_action(states, actions)
    
        # 比率 (r_t(θ)) を計算
        ratios = torch.exp(new_log_probs - log_probs)
    
        # PPOの損失関数を計算
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
    
        # 値関数の損失
        critic_loss = F.mse_loss(values, returns)
    
        # エントロピー正則化
        entropy_loss = -self.ent_coef * entropy.mean()
    
        # 合計損失
        loss = actor_loss + 0.5 * critic_loss + entropy_loss
    
        # パラメータの更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
    def train(self, env, max_steps=100, epochs=10, EPISODES=50, batch_size=64, LOG_DIR=None):
        if LOG_DIR:
            summary_dir = LOG_DIR + '/test_run_' + datetime.now().strftime('%m%d_%H%M')
            summary_writer = SummaryWriter(log_dir=summary_dir)
    
        for episode in range(EPISODES):
            state = env.reset()
            memory = {'states': [], 'actions': [], 'rewards': [], 'log_probs': [], 'values': [], 'dones': []}
            episode_reward = 0
    
            for step in range(max_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action, log_prob, _ = self.actor_critic.get_action(state_tensor)
                value = self.actor_critic.get_value(state_tensor).item()
                next_state, reward, done = env.step(action)
    
                memory['states'].append(state)
                memory['actions'].append(action)
                memory['rewards'].append(reward)
                memory['log_probs'].append(log_prob)
                memory['dones'].append(done)
                memory['values'].append(value)
    
                state = next_state
                episode_reward += reward
    
                if done:
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                    next_value = self.actor_critic.get_value(next_state_tensor).item()
    
                    states = torch.tensor(np.array(memory['states']), dtype=torch.float32)
                    actions = torch.tensor(memory['actions'], dtype=torch.float32 if self.continuous else torch.long)
                    log_probs = torch.stack(memory['log_probs'])
                    rewards = torch.tensor(memory['rewards'], dtype=torch.float32)
                    dones = torch.tensor(memory['dones'], dtype=torch.float32)
                    values = torch.tensor(memory['values'], dtype=torch.float32)
    
                    returns, advantages = self.compute_gae(rewards, values, next_value, dones)
                    summary_writer.add_scalar("Episode Reward", episode_reward, episode)
                    summary_writer.add_scalar("value", values[-1], episode)
                    
                    print(f"Episode {episode + 1}: Reward : {episode_reward}")
                    print(f"Episode {episode + 1}: values : {values[-1]}")
                    
                    if episode % epochs == 0:
                        self.ppo_update(states, actions, log_probs, returns, advantages)

                    break  # エピソード終了
 