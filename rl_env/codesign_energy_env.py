import numpy as np
import pandas as pd

class CodesignEnergyEnv:
    def __init__(self,mode = 'discrete'):
        self.mode = mode
        
        # データの読み込み
        self.solar_radiation_all = np.load("C:/Users/manta/OneDrive/ドキュメント/IEEE/data/sample_data_pv.npy") 
        self.solar_radiation = self.solar_radiation_all[4345:4345 + 24*7]
        self.market = pd.read_csv("C:/Users/manta/OneDrive/ドキュメント/IEEE/data/spot_summary_2022.csv", encoding='shift_jis')
        self.market_prices = self.extract_market_prices(self.market,8690,8690 + 24*2*7)
        
        # 蓄電池のパラメータ
        self.eta_c_t = 0.96  # 充電効率
        self.eta_d_t = 0.995  # 放電効率
        self.soc_max = 0.9
        self.soc_min = 0.1 
        
        # 学習の設定
        self.scaling_values = np.array([0.0,0.5,1.0])  # scaling値を0.0, 0.5, 1.0に設定
        self.bidding_values = np.arange(0.0, 0.7, 0.1)  # bidding値を0.0から0.8まで0.1刻みで設定
        
        self.observation_space = 3 # 充放電の状態, 日射量, 市場価格をobservationとする
        self.action_space = 2 if mode == "continuous" else len(self.scaling_values) * len(self.bidding_values)
        self.rho_space    = 1
        
    def extract_market_prices(self, market_df, start_row, end_row):
        """6列目のデータを抽出し、エピソードごとに使用"""
        market_prices = market_df.iloc[start_row:end_row, 5].astype(float).dropna().values
        prices_per_episode = []
        for i in range(0, len(market_prices) - 1, 2):
            episode_prices = market_prices[i:i+2]
            prices_per_episode.append(episode_prices)
        return np.array(prices_per_episode)

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

    def step(self, action,rho):
        # アクションからscalingとbiddingを決定
        if self.mode == "discrete":
            scaling_idx = action // len(self.bidding_values)
            bidding_idx = action % len(self.bidding_values)
            
            self.scaling = self.scaling_values[scaling_idx]
            self.bidding = self.bidding_values[bidding_idx]
        else:
            # 連続アクションの場合
            self.scaling = np.clip(action[0], 0.0, 1.0)
            self.bidding = np.clip(action[1]*0.6, 0.0, 0.6)

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
        
        if self.current_step >= len(self.solar_radiation):
            done = True
            observation = np.array([self.soc, 0.0, 0.0], dtype=np.float32)  # 終了時のダミー観測値
        else:
            observation = np.array([self.soc, self.solar_radiation[self.current_step], current_market_price], dtype=np.float32)
                
        reward = self.calculate_reward(self.soc, current_market_price, current_solar_radiation, self.bidding, self.Pc_t, self.Pd_t, delta_t) 
        
        self.step_rewards.append(reward)
        
        return observation, reward, done

    def calculate_reward(self, soc, market_prices, solar_generation, bidding, Pc_t, Pd_t, delta_t):
        lambda_t = market_prices
        xD_t = solar_generation - Pc_t + Pd_t
        rho_pen = 1.0
        self.penalty = rho_pen * abs(bidding - xD_t)
        self.penalty_history.append(self.penalty)
        beta_t = 1.0
        ft = (lambda_t * (xD_t - self.penalty) - beta_t * (Pc_t + Pd_t)) * delta_t
        reward = ft - lambda_t * solar_generation * delta_t
        return reward
    
    # def calculate_reward(self, soc, market_prices, solar_radiation, bidding, Pc_t, Pd_t, delta_t):

    #     error = solar_radiation - bidding
    #     reward = - (error ** 2)
    #     return reward
