import numpy as np
import pandas as pd

class AllCodesignEnergyEnv:
    def __init__(self,agent_idx):
        # データの読み込み
        self.solar_radiation_all = np.load("/home/students3/mantani/IEEE/data/sample_data_pv.npy") 
        self.market_all = pd.read_csv("/home/students3/mantani/IEEE/data/spot_summary_2022.csv", encoding='shift_jis')
        
        self.days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.hours_per_day = 24
        
        # 蓄電池のパラメータ
        self.eta_c_t = 0.96  # 充電効率
        self.eta_d_t = 0.995  # 放電効率
        self.soc_max = 0.9
        self.soc_min = 0.1 
        
        self.scaling_values = np.array([0.0,0.5,1.0,1.5])  # scaling値を0.0, 0.5, 1.0に設定
        self.bidding_values = np.arange(0.0, 0.9, 0.1)  # bidding値を0.0から0.8まで0.1刻みで設定
        self.observation_space = 3 # 充放電の状態, 日射量, 市場価格をobservationとする
        self.action_space = len(self.scaling_values) * len(self.bidding_values) # アクションの次元の設定
        
        #データを1か月ごとに分割
        
        self.agent_idx = agent_idx
        
        self.month_index = self.calculate_month_index(self.days_per_month, self.hours_per_day)
        
        if self.agent_idx == 'all' :
            start_idx,end_idx = [0,8760]
        else:
            start_idx, end_idx = self.month_index[self.agent_idx]
        
        self.solar_radiation = self.solar_radiation_all[start_idx:end_idx]
        self.market_prices = self.extract_market_prices(self.market_all, start_idx, end_idx)
        
        self.penalty_history = []
        self.battery_penalty_history = []
   
    def calculate_month_index(self, days_per_month, hours_per_day):
        indices = []
        start_idx = 0
        for days in days_per_month:
            end_idx = start_idx + days * hours_per_day
            indices.append((start_idx, end_idx))
            start_idx = end_idx
        return indices
    
    def extract_market_prices(self, market_df, start_row, end_row):
        market_prices = market_df.iloc[2*start_row:2*end_row, 5].astype(float).dropna().values
        prices_per_episode = []
        for i in range(0, len(market_prices) - 1, 2):
            episode_prices = market_prices[i:i+2]
            prices_per_episode.append(episode_prices)
        return np.array(prices_per_episode)

        # 蓄電池のパラメータ
        self.eta_c_t = 0.96  # 充電効率
        self.eta_d_t = 0.995  # 放電効率
        self.soc_max = 0.9
        self.soc_min = 0.1 
         
        self.observation_space = 3 # 充放電の状態, 日射量, 市場価格をobservationとする
        self.action_space = len(self.scaling_values) * len(self.bidding_values) # アクションの次元の設定
        
    def reset(self, rho):
        # パラメータの設定
        self.E_max = rho
        self.P_max = rho
        
        #初期化 
        self.soc = 0.5  
        self.current_step = 0
        self.total_reward = 0.0  # 累積報酬の初期化
        
        # 保存用
        self.soc_history = [] 
        self.solar_generation_history = []
        self.step_rewards = []
        self.Pc_t = []
        self.Pd_t = []
        self.penalty_history = []
        
        observation = np.array([self.soc, self.solar_radiation[self.current_step], self.market_prices[self.current_step, 0]], dtype=np.float32)
        return observation

    def step(self, action):
        # アクションからscalingとbiddingを決定
        scaling_idx =  action // len(self.bidding_values)
        bidding_idx = action % len(self.bidding_values)
        
        self.scaling = self.scaling_values[scaling_idx]
        self.bidding = self.bidding_values[bidding_idx]
        # self.rho = self.rho_values[rho_idx]
        
        delta_t = 0.25
        self.soc_history.append(self.soc)
        
        done = False
        
        # 現在の時刻のsolar_radiationと現在のエピソードの市場価格を取得
        current_solar_radiation = self.solar_radiation[self.current_step] 
        current_market_price = np.mean(self.market_prices[self.current_step], axis=0)

        self.solar_generation_history.append(current_solar_radiation)

        self.Pcmax_t = self.E_max * (self.soc_max - self.soc) / delta_t
        self.Pdmax_t = self.E_max * (self.soc - self.soc_min) / delta_t
        
        # SoCがsoc_maxを超えないように充電量を制限
        self.Pbarc_t = max(0,min(self.P_max,  self.Pcmax_t))
        self.Pbard_t = max(0,min(self.P_max,  self.Pdmax_t))   
        
        if current_solar_radiation > self.bidding:
            self.Pc_t = min(self.scaling * (current_solar_radiation - self.bidding), self.Pbarc_t)
            self.Pd_t = 0
        else:
            self.Pd_t = min(self.scaling * (self.bidding - current_solar_radiation), self.Pbard_t)
            self.Pc_t = 0
            
        self.soc += (self.eta_c_t * self.Pc_t / self.E_max) * delta_t - ((1 / self.eta_d_t) * (self.Pd_t / self.E_max)) * delta_t
        # self.soc = np.clip(self.soc, self.soc_min, self.soc_max)
           
        self.current_step += 1
        
        if self.current_step >= len(self.solar_radiation) or self.current_step >= len(self.market_prices):
            done = True
            observation = np.array([self.soc, 0.0, 0.0], dtype=np.float32)  # 終了時のダミー観測値
        else:
            current_solar_radiation = self.solar_radiation[self.current_step] 
            current_market_price = np.mean(self.market_prices[self.current_step], axis=0)
            observation = np.array([self.soc, self.solar_radiation[self.current_step], current_market_price], dtype=np.float32)
                
        reward = self.calculate_reward(self.agent_idx, self.days_per_month, self.soc, current_market_price, current_solar_radiation, self.bidding, self.Pc_t, self.Pd_t, delta_t) 
        
        self.step_rewards.append(reward)
        
        return observation, reward, done

    def calculate_reward(self, agent_idx, days_per_month, soc, market_prices, solar_generation, bidding, Pc_t, Pd_t, delta_t):
        lambda_t = market_prices
        xD_t = solar_generation - Pc_t + Pd_t
        rho_pen = 0.5
        self.deviation_penalty = rho_pen * abs(bidding - xD_t)
        self.penalty = rho_pen * abs(bidding - xD_t) * lambda_t*delta_t
        self.penalty_history.append(self.penalty)
        beta_t = 1.0
        self.battery_penalty = beta_t * (Pc_t + Pd_t)*delta_t
        self.battery_penalty_history.append(self.battery_penalty)
        
        ft = (lambda_t * (xD_t - self.deviation_penalty) - beta_t * (Pc_t + Pd_t)) * delta_t
        
        # reward = (ft - lambda_t * solar_generation * delta_t)
        if agent_idx == 'all':
           reward = ft - lambda_t * solar_generation * delta_t
        else:
           reward = (ft - lambda_t * solar_generation * delta_t) * 365 /  days_per_month[agent_idx]  
        return reward
