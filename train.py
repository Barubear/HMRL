
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results,ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

import numpy as np


def train(model,env,total_timesteps, save_path,log_path,test_times,testonly =False):
    
    if not testonly:
        msg_pre_tarin =evaluate_policy(model,env,n_eval_episodes=10,deterministic=True)
        #print(env)
        model.learn(total_timesteps, log_interval=4,progress_bar=True,callback = SaceBaseCallback(save_path,log_path))
        
        
        

    print('model test:')

    state_value_list = []
    model = RecurrentPPO.load(save_path)
    
    
    for i in range(test_times):
        step = 0
        obs= env.reset()
        while True:

                action, _states = model.predict(obs)
                obs, rewards, dones, info  = env.step(action)
                
                step +=1
                if dones :
                    print(step/6)
                    break
                elif step >= 10000:
                    
                    break
    
    

class SaceBaseCallback(BaseCallback):
    def __init__(self, save_path,log_path):
        super().__init__(verbose=0)
        self.best = -float('inf')
        self.save_path = save_path
        self.log_path = log_path
        self.best_step = 0
    def _on_step(self) -> bool:
        
        if self.n_calls%1000 != 0:
            
            return True
        x , y = ts2xy(load_results(self.log_path),'timesteps')
        mean_reward = sum(y[-100:])/len(y[-100:])
        print(self.best_step)
        if mean_reward >self.best:
            self.best = mean_reward
            self.best_step = self.n_calls
            print(self.n_calls,self.best)
            self.model.save(self.save_path)
        
        return True
    
