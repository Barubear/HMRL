from typing import List
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
import numpy as np
import random
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
import math

from Envs.navi_Env import navi_Env
from Envs.Enemy import Enemy

class shoot_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

     
     
    def __init__(self,render_mode = "human",width = 15,height = 15):
        super().__init__()
        self.width = width
        self.height = height
        


        self.max_HP = 10 
        self.curr_HP = self.max_HP
        self.agent_pos = self.random_Agent_pos().copy()
        
        self.enemy_pos = self.get_enemy().copy()

        self.navi = RecurrentPPO.load("save_path")
        self.navi_env = make_vec_env('navi_Env-v0')

        self.navi_state = 0 #0:未启动，1：执行中，2：已到达

        self.killed = False

        self.max_skill_dis = 7
        
        

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0,15,(2 ,),np.int32),
            "enemy": spaces.Box(0,15,(2 ,),np.int32),
            'HP': spaces.Discrete(self.max_HP+1) ,
            "navi":spaces.Discrete(3)
            })

        
        
            
        self.action_space = spaces.MultiDiscrete([2, 2])

    def reset(self,seed=None, options=None):
        self.curr_HP = self.max_HP
        self.agent_pos = self.random_Agent_pos().copy()
        
        self.enemy_pos = self.get_enemy().copy()

        self.navi_state = 0 #0:未启动，1：执行中，2：已到达

        self.killed = False

        return self._get_obs() , self._get_info()

    def _get_obs(self):
        return {
            "agent": np.array(self.agent_pos, dtype=np.int32),
            'enemy': np.array(self.enemy_pos, dtype=np.int32),
            'HP': self.curr_HP,
            "navi":self.navi_state
        }
    
    def _get_info(self):
        
        return {
            "agent": self.agent_pos,
            'enemy': self.enemy_pos,
            'HP': self.curr_HP,
            "navi":self.navi_state
            
        }



    def step(self, action):
        reward = 0
        action_type = action[0]
        action_dic = action[1]
        terminated = False

        if action_type == 0:# use navi 
            
            if self.navi_state == 0:
                self.navi_state = 1

            reward -= 1

            navi_action, _states = self.navi.predict(self.get_navi_obs(action_dic))
            _obs, rewards, dones, _info  = self.navi_env.step(action)

            

            next_x= self.agent_pos[0]
            next_y =self.agent_pos[1]

            if(navi_action == 0):#up
                next_y-=1
            elif(navi_action == 1):#down
                next_y+=1
            elif(navi_action == 2):#right
                next_x+=1
            elif(navi_action == 3):#left
                next_x-=1
            
            if(next_x < 0 or next_x >=self.width or next_y < 0 or next_y >=self.height):
                reward -= 1
            elif(next_x== self.enemy_pos[0] and next_y == self.enemy_pos[1]) :
                reward -=10
                self.curr_HP-=1
                if(self.curr_HP <= 0):
                    reward -= 50
                    terminated = True
            else:
                self._update_agent_position(next_x,next_y)

            if _info[0]["arrive"]:
                self.navi_state = 2
                
                    
        elif (action_type == 1):#attack
            
            if self.navi_state == 1:
                reward -=10
                self.navi_state = 0

            skill_dis = self.get_distance(action_dic,self.agent_pos)
            if skill_dis <=  self.max_skill_dis:

                dis = self.get_distance(action_dic,self.enemy_pos)

                if dis < 2:
                    reward += 100
                else:
                    reward -=15    

            else:
                reward -= 15
        
        observation = self._get_obs()
        info = self._get_info()
        print(action,info)
        return observation, reward, terminated, False, info
    
    def get_navi_obs(self,pos):
        return {
            "agent": np.array(self.agent_pos, dtype=np.int32),
            "goal": np.array(pos, dtype=np.int32),
        }



    def _update_agent_position(self, next_x,next_y):

        self.agent_pos = [next_x, next_y]
    
    
    
    def get_enemy(self):
        enemy = [-1,-1]
        while True:
            x = random.randint(0, 14)
            y = random.randint(0, 14)
            if self.get_distance(self.agent_pos,[x,y] )>= 9:
                enemy = [x,y]
                break
        return enemy
            



    def random_Agent_pos(self):
        x = random.randint(0, 14)
        y = random.randint(0, 14)
        return [x,y]

    def get_distance(self,pos1,pos2):
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return distance




register(
    id='shoot_Env-v0',
    entry_point='Envs.shoot_Env:shoot_Env',
    max_episode_steps=1000,
)