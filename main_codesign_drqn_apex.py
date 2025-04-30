import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import ray
import pandas as pd

# Environment
from rl_env.codesign_energy_year_env import AllCodesignEnergyEnv

###############################################################################
# Main func with pytorch
from agent.codesign_drqn_ApeX import Q_net, EpisodeMemory, EpisodeBuffer, train

@ray.remote
class Worker:
    def __init__(self,  observation_space, action_space, rho_space, device, agent_idx):
        self.agent_idx = agent_idx
        self.device = device    
        self.env = AllCodesignEnergyEnv(agent_idx)
        
        self.Q        = Q_net(observation_space, action_space, rho_space).to(device)
        self.Q_target = Q_net(observation_space, action_space, rho_space).to(device)
                
    def run_episode(self, epsilon, gamma, max_step, current_weights, mu, sigma):
        self.Q.load_state_dict(current_weights)
        rho = max(0.05, np.random.normal(mu, sigma))
        obs = self.env.reset(rho)
        h, c = self.Q.init_hidden_state(batch_size=1, training=False)
        episode_reward = 0
        episode_reward_discount = 0
        episode_record = EpisodeBuffer()
        
        for t in range(max_step):
            obs, h, c, episode_reward, episode_record, episode_reward_discount, done = self.simulate_train_step(obs, rho, h, c, epsilon, gamma, episode_reward, episode_record,  episode_reward_discount)
            
            if done:
                break

        return self.agent_idx, rho, episode_reward, episode_record
    
    def simulate_train_step(self,  obs, rho, h, c, epsilon, gamma, episode_reward, episode_record, episode_reward_discount):
        self.episode_q_values = []
        # 観測とrhoを結合して行動を取得
        obs_rho = np.concatenate([obs, np.array([rho])])  # [obs, rho].numpy
        obs_rho = torch.from_numpy(obs_rho).float().to(self.device).unsqueeze(0).unsqueeze(0)  # [[[obs,rho]]].tensor

        # 行動をサンプリング
        a, h, c = self.Q.sample_action(obs_rho, h.to(self.device), c.to(self.device), epsilon, np.array([rho]))

        # 環境を一歩進める
        obs_prime, r, done = self.env.step(a)
        
        # 報酬を更新
        episode_reward += r
        episode_reward_discount = r + gamma * episode_reward_discount

        # リプレイバッファに (s, a, r, s') を追加
        done_mask = 0.0 if done else 1.0
        episode_record.put([obs, a, r, obs_prime, done_mask, rho])

        # 次の状態に更新
        obs = obs_prime    
        
        # Q値を計算して記録
        with torch.no_grad():
            q_values, _, _ = self.Q(obs_rho, h, c)
            max_q_value = q_values.max().item()
            self.episode_q_values.append(max_q_value)
            
        return obs, h, c, episode_reward, episode_record, episode_reward_discount, done
    
@ray.remote(num_cpus = 1)
class Learner:

    def __init__(self,observation_space, action_space, rho_space, device, 
                 tau, target_update_period, 
                 learning_rate=1e-4, learning_rate_mu=1e-4, learning_rate_sigma=0, retrain = None):
        self.device = device
        self.tau = tau
        self.target_update_period = target_update_period
        
        self.Q        = Q_net(observation_space, action_space, rho_space).to(device)
        self.Q_target = Q_net(observation_space, action_space, rho_space).to(device)
        if retrain:
             self.Q.load_state_dict(torch.load("saved_weights.pth"))
             self.Q_target.load_state_dict(torch.load("saved_weights.pth"))
        self.learning_rate_mu    = learning_rate_mu
        self.learning_rate_sigma = learning_rate_sigma

        self.episode_q_values = []
        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.battery_price = 1000 
        
    def get_weights(self):
        return self.Q.state_dict()
    
    def learn(self, minibatches, batch_size, min_epi_num, update_cycles):
        
          t =0 
          for samples, seq_len in minibatches:
              t += 1
              # DRQNの学習ステップ
              train(self.Q, self.Q_target, samples, seq_len, self.device,
                      optimizer=self.optimizer,
                      batch_size=batch_size)
                    
              # ターットネットワークの更新
              if (t+1) % self.target_update_period == 0:        
                    for target_param, local_param in zip(self.Q_target.parameters(), self.Q.parameters()):
                        target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                    
          return self.Q.state_dict()

    def update_phi(self, mu, sigma, rho_list, income_list):
        # list to numpy
        rhos    = np.array(rho_list)
        incomes = np.array(income_list)
        # compute G
        G = incomes - rhos * self.battery_price
        
        # muの更新
        mu_grad = (((rhos - mu) / (sigma ** 2)) * (G - G.mean())).mean()
        mu += self.learning_rate_mu * mu_grad
        
        # sigmaの更新 
        sigma_grad = 0
        sigma += self.learning_rate_sigma * sigma_grad
        
        # 更新後のphiを返す
        phi = [mu, sigma]
        
        return phi
    
    def save_weights(self,ckpt_path):
        
        torch.save({'Q_net': self.Q.state_dict(),
                    'target-Q_net': self.Q_target.state_dict(),
                    }, 
        ckpt_path)

    
@ray.remote  
class Tester:
    def __init__(self,  observation_space, action_space, rho_space, device, agent_idx ,epsilon_tester):
        self.agent_idx = agent_idx
        self.device = device    
        self.env = AllCodesignEnergyEnv(agent_idx)
        self.epsilon_tester  = epsilon_tester
        self.Q        = Q_net(observation_space, action_space, rho_space).to(device)
        self.Q_target = Q_net(observation_space, action_space, rho_space).to(device)
                
    def run_episode(self, epsilon_tester, gamma, max_step, current_weights, mu, sigma):
        self.Q.load_state_dict(current_weights)
        rho = max(0.05, np.random.normal(mu, sigma))
        obs = self.env.reset(rho)
        h, c = self.Q.init_hidden_state(batch_size=1, training=False)
        episode_reward = 0
        episode_reward_discount = 0        
        episode_penalty_history = []
        episode_battery_penalty_history = []
        episode_bidding_history = []
        episode_record = EpisodeBuffer()
        
        for t in range(max_step):
            obs, h, c, episode_reward, episode_record, episode_reward_discount, done, episode_penalty_history, episode_battery_penalty_history,episode_bidding_history  = self.simulate_train_step(obs, rho, h, c, epsilon_tester, gamma, episode_reward, episode_record,  episode_reward_discount, episode_penalty_history, episode_battery_penalty_history,episode_bidding_history)
            
            if done:
                break
    
        return self.agent_idx, rho, episode_reward, episode_record, episode_penalty_history, episode_battery_penalty_history,episode_bidding_history
    
    def simulate_train_step(self,  obs, rho, h, c, epsilon_tester, gamma, episode_reward, episode_record, episode_reward_discount, episode_penalty_history, episode_battery_penalty_history,episode_bidding_history):
        self.episode_q_values = []
        # 観測とrhoを結合して行動を取得
        obs_rho = np.concatenate([obs, np.array([rho])])  # [obs, rho].numpy
        obs_rho = torch.from_numpy(obs_rho).float().to(self.device).unsqueeze(0).unsqueeze(0)  # [[[obs,rho]]].tensor
    
        # 行動をサンプリング
        a, h, c = self.Q.sample_action(obs_rho, h.to(self.device), c.to(self.device), epsilon_tester, np.array([rho]))
    
        # 環境を一歩進める
        obs_prime, r, done = self.env.step(a)
        
        # 報酬を更新
        episode_reward += r
        episode_reward_discount = r + gamma * episode_reward_discount
        
        episode_bidding_history.append(self.env.bidding)
        episode_penalty_history.append(self.env.penalty)
        episode_battery_penalty_history.append(self.env.battery_penalty)
        
        # リプレイバッファに (s, a, r, s') を追加
        done_mask = 0.0 if done else 1.0
        episode_record.put([obs, a, r, obs_prime, done_mask, rho])
    
        # 次の状態に更新
        obs = obs_prime    
        
        # Q値を計算して記録
        with torch.no_grad():
            q_values, _, _ = self.Q(obs_rho, h, c)
            max_q_value = q_values.max().item()
            self.episode_q_values.append(max_q_value)
            
        return obs, h, c, episode_reward, episode_record, episode_reward_discount, done, episode_penalty_history, episode_battery_penalty_history,episode_bidding_history

####################################################################
# Main training loop
####################################################################
def main(num_cpus=12, gamma=0.99):
    random.seed(1)
    np.random.seed(1)
    
    # Environment setting
    ##########################################
    env = AllCodesignEnergyEnv(0)
    observation_space = env.observation_space
    action_space      = env.action_space
    rho_space         = 1 

    ############################################
    # DRQN setting parameters
    ############################################
    batch_size = 8
    learning_rate = 1e-3
    min_epi_num  = 20 # Start moment to train the Q network
    num_updates  = 20000
    print_per_iter = 30
    target_update_period = 4
    eps_start = 1.0
    eps_end   = 0.001
    eps_decay = 0.9995
    tau       = 1e-2
    max_step  = 8760
    max_epi_num = 100
    
    # DRQN param
    random_update = True # If you want to do random update instead of sequential update
    lookup_step   = 24 * 1 # If you want to do random update instead of sequential update
    max_epi_len   = 8760

    ##############################
    # Codesing parameters
    ##############################
    learning_rate_mu    = 1e-5
    learning_rate_sigma = 0
    min_epi_codesign =  2000
    device = torch.device("cpu")

    ##############################
    # Initalization
    ##############################
    episode_memory = EpisodeMemory(random_update=random_update, 
                                   max_epi_num=max_epi_num, max_epi_len=max_epi_len, 
                                   batch_size=batch_size, 
                                   lookup_step=lookup_step)

    # data for co-design
    mu    = 0.6
    sigma = 0.2
    income_list = [] ############# TODO: reward -> income
    rho_list    = []
    bidding_history = []  # List to collect bidding data
    history =[]
    log_dir = 'codesign_apex_logs/test_run_'+datetime.now().strftime('%m%d%H%M')
    writers  = SummaryWriter(log_dir=log_dir)
    
    ######################################################################################
    # 並列処理の準備
    ##################################################################################
    ray.init(ignore_reinit_error=True)      
    
    workers = [Worker.remote(observation_space, action_space, rho_space, device, agent_idx) for agent_idx in range(num_cpus)]
    learner = Learner.remote(observation_space, action_space, rho_space, device, tau, target_update_period,
                             learning_rate, learning_rate_mu, learning_rate_sigma)
    epsilon = eps_start
    epsilon_tester = 0
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    current_weights = ray.get(learner.get_weights.remote())    
    current_weights = ray.put(current_weights)
    
    tester = Tester.remote(observation_space, action_space, rho_space, device, 'all',  epsilon_tester)
    
    # Workerプロセスの開始
    worker_results = [worker.run_episode.remote(epsilon,gamma,max_step,current_weights, mu,sigma)       for worker in workers]      
    
    # まず30エピソード分データを蓄積
    for _ in range(30):
        finished, worker_results = ray.wait(worker_results, num_returns = 1)  
        agent_idx, rho,episode_reward, episode_record = ray.get(finished[0])
        episode_memory.put(episode_record)
        worker_results.extend([workers[agent_idx].run_episode.remote(epsilon,gamma,max_step, current_weights, mu,sigma)] )        

    #　Learnerの学習を開始       
    minibatches = [episode_memory.sample() for _ in range(24*7)]
    wip_learner = learner.learn.remote(minibatches, batch_size, min_epi_num, 0)
    wip_tester = tester.run_episode.remote(epsilon_tester,gamma,max_step, current_weights, mu,sigma)
    
    # 反復を開始        
    update_cycles = 1
    actor_cycles = 0
    while update_cycles <= num_updates:
        # workerによるデータの収集
        actor_cycles += 1
        finished, worker_results = ray.wait(worker_results, num_returns=1)
        agent_idx, rho, episode_reward, episode_record = ray.get(finished[0])
        episode_memory.put(episode_record)
        
        income_list.append(episode_reward)
        rho_list.append(rho)
        # print(f'agent_idx: {agent_idx+1}, rho_list[agent_idx]:{len(rho_list[agent_idx])}.income_list[agent_idx]:{len(income_list[agent_idx])}')
        #print(f'agent_idx: {agent_idx+1}, episode: {episode + 1}, episode_reward: {episode_reward}')
        #writers.add_scalar('Rewards', episode_reward, update_cycles)
        #writers.add_scalar('Mu', mu, update_cycles)
        
        worker_results.extend([workers[agent_idx].run_episode.remote(epsilon, gamma, max_step, current_weights, mu, sigma)])   

        # Learnerのタスク完了判定とDRQNの更新
        finished_learner, _ = ray.wait([wip_learner], timeout=0)
        if finished_learner:
            update_cycles += 1
            print("Actorが遷移をReplayに渡した回数：", actor_cycles)
            actor_cycles = 0

            # DRQNの更新の開始
            new_weights     = ray.get(finished_learner[0])
            current_weights = ray.put(new_weights)
            
            minibatches = [episode_memory.sample() for _ in range(24*7)]
            wip_learner     = learner.learn.remote(minibatches,  batch_size, min_epi_num, update_cycles)
            
            if update_cycles % 5 == 0:
                test_score = ray.get(wip_tester)
                print(update_cycles, test_score[1], test_score[2],sum(test_score[4]),sum(test_score[5]))
                history.append((update_cycles-5, test_score))
                wip_tester = tester.run_episode.remote(epsilon_tester,gamma,max_step, current_weights, mu,sigma)
                rho = test_score[1]
                episode_reward = test_score[2]  
                episode_bidding = test_score[6]
                writers.add_scalar('wip_tester/rho', rho, update_cycles)
                writers.add_scalar('wip_tester/rewards', episode_reward, update_cycles)
                writers.add_scalar('wip_tester/Deviation_penalty', sum(test_score[4]), update_cycles)
                writers.add_scalar('wip_tester/Degradation', sum(test_score[5]), update_cycles)
                bidding_history.append(episode_bidding)
                if update_cycles % num_updates == 0:
                    bidding_df = pd.DataFrame(episode_bidding)
                    bidding_df.to_csv(f'drqn_apex_action/episode_bidding_battery_{current_time}.csv',index=False)
                
            # # Phiの更新
            if update_cycles > min_epi_codesign:
                if update_cycles % 10 == 0:
                    phi = ray.get( learner.update_phi.remote(mu, sigma, rho_list, income_list))
                    mu,sigma = phi
                    writers.add_scalar('phi/mu', mu, update_cycles)
                    rho_list = []
                    income_list = []
                    print(f'Updated phi: {mu}')

            # update epsilon
            epsilon = max(eps_end, epsilon * eps_decay) # Linear annealing

            writers.add_scalar('epsilon', epsilon, update_cycles)

            
            # penalty_history.append(episode_penalty_history)
            # battery_penalty_history.append(episode_battery_penalty_history)
            #save agent
            if update_cycles % 5000 == 0:
                ckpt_path = log_dir + f'/agent-{update_cycles}.pth'
                learner.save_weights.remote(ckpt_path)
                               
    ray.shutdown()

if __name__ == "__main__":
     main()
     