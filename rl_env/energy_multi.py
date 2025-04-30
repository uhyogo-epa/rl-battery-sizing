import numpy as np
import pandas as pd

class RenewableEnergyEnv:
    def __init__(self,battery_times):
        self.eta_c_t = 0.96  # 充電効率
        self.eta_d_t = 0.995  # 放電効率
        self.observation_space = 3  # 充放電の状態, 日射量, 市場価格をobservationとする
        self.scaling_values = np.array([0.0,0.5,1.0])  # scaling値を0.0, 0.5, 1.0に設定
        self.bidding_values = np.arange(0.0, 0.7, 0.1)  # bidding値を0.0から0.7まで0.1刻みで設定
        self.action_space = len(self.scaling_values) * len(self.bidding_values)  # アクションの次元の設定
        self.current_step = 0
        self.soc_max = 0.9
        self.soc_min = 0.1
        self.solar_radiation_all = np.load("C:/Users/manta/OneDrive/ドキュメント/IEEE/data/sample_data_pv.npy") 
        self.solar_radiation = self.solar_radiation_all[4345:4345 + 24*7]
        self.market = pd.read_csv("C:/Users/manta/OneDrive/ドキュメント/IEEE/data/spot_summary_2022.csv", encoding='shift_jis')
        self.market_prices = self.extract_market_prices(self.market,4370,4370 + 24*2*7)
        self.gamma = 0.9 # 割引率の設定
        self.battery_times = battery_times
        self.E_max = max(self.solar_radiation) * self.battery_times
        self.P_max = max(self.solar_radiation) * self.battery_times
        self.step_rewards = []        
        self.penalty_history = []
        self.battery_penalty_history = []
        self.xD_t_history = []
        self.reset()

    def extract_market_prices(self, market_df, start_row, end_row):
        """6列目のデータを抽出し、エピソードごとに使用"""
        market_prices = market_df.iloc[start_row:end_row, 5].astype(float).dropna().values
        prices_per_episode = []
        for i in range(0, len(market_prices) - 1, 2):
            episode_prices = market_prices[i:i+2]
            prices_per_episode.append(episode_prices)
        return np.array(prices_per_episode)

    def reset(self):
        self.soc = 0.5  
        self.soc_history = [] 
        self.bidding = 0.0
        self.scaling = 1.0
        self.current_step = 0
        self.current_episode = 0
        self.total_reward = 0.0  # 累積報酬の初期化
        self.solar_generation_history = []
        self.step_rewards = []
        self.Pc_t = []
        self.Pd_t = []
        self.penalty_history = []
        self.battery_penalty_history = []
        self.xD_t_history = []
        
        # 初期観測値を設定
        observation = np.array([self.soc, self.solar_radiation[self.current_step], self.market_prices[self.current_episode, 0]], dtype=np.float32)
        return observation    

    def step(self, action):
        # アクションからscalingとbiddingを決定
        
        scaling_idx = action // len(self.bidding_values)
        bidding_idx = action % len(self.bidding_values)
        
        self.scaling = self.scaling_values[scaling_idx]
        self.bidding = self.bidding_values[bidding_idx]
        
        self.bidding = np.clip(self.bidding, 0.0, 0.8)  
        self.scaling = np.clip(self.scaling, 0.0, 1.0)
        delta_t = 0.25
        self.soc_history.append(self.soc)
        
        done = False
        
        # 現在の時刻のsolar_radiationと現在のエピソードの市場価格を取得
        current_solar_radiation = self.solar_radiation[self.current_step] 
        current_market_price = np.mean(self.market_prices[self.current_episode], axis=0)

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
        
        self.current_episode += 1

        if self.current_step >= len(self.solar_radiation):
            done = True
            observation = np.array([self.soc, 0.0, 0.0], dtype=np.float32)  # 終了時のダミー観測値
        
        else:
            observation = np.array([self.soc, self.solar_radiation[self.current_step], current_market_price], dtype=np.float32)
                
        reward = self.calculate_reward(self.soc, current_market_price, current_solar_radiation, self.bidding, self.Pc_t, self.Pd_t, delta_t) 
        
        self.total_reward = self.total_reward * self.gamma + reward  # 割引率を適用
        
        self.step_rewards.append(reward)

        return observation, reward, done

    def calculate_reward(self, soc, market_prices, solar_generation, bidding, Pc_t, Pd_t, delta_t):
        lambda_t = market_prices
        self.xD_t = solar_generation - Pc_t + Pd_t
        self.xD_t_history.append(self.xD_t)
        
        rho_pen = 1.0
        self.penalty = rho_pen * abs(bidding - self.xD_t)
        self.penalty_history.append(self.penalty)
        
        beta_t = 1.0
        self.battery_penalty = beta_t * (Pc_t + Pd_t)
        self.battery_penalty_history.append(self.battery_penalty)
        
        ft = (lambda_t * (self.xD_t - self.penalty) - beta_t * (Pc_t + Pd_t)) * delta_t
        reward = ft - lambda_t * solar_generation * delta_t
        return reward
    
    # def calculate_reward(self, soc, market_prices, solar_radiation, bidding, Pc_t, Pd_t, delta_t):

    #     error = solar_radiation - bidding
    #     reward = - (error ** 2)
    #     return reward
