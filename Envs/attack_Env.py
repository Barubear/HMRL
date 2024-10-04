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

class attack_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

     
     
    def __init__(self,render_mode = "human",width = 15,height = 15):
        super().__init__()
        self.width = width
        self.height = height
        


        self.max_HP = 10 
        self.curr_HP = self.max_HP
        self.agent_pos = self.random_Agent_pos().copy()
        
        self.enemy = self.get_enemy().copy()
        self.enemy_pos = [self.enemy[0],self.enemy[1]]

        self.navi = RecurrentPPO.load('trained_modules/navie/normal_best')
    
        self.navi_state = 0 #0:未启动，1：执行中，2：已到达

        self.killed = False

        self.max_skill_dis = 7
        
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0,15,(2 ,),np.int32),
            "enemy": spaces.Box(0,15,(3 ,),np.int32),
            'HP': spaces.Discrete(self.max_HP+1) ,
            "navi_state":spaces.Discrete(3)
            })

        
        
            
        self.action_space = spaces.MultiDiscrete([2, 15, 15])

    def reset(self,seed=None, options=None):
        self.curr_HP = self.max_HP
        self.agent_pos = self.random_Agent_pos().copy()
        
        self.enemy = self.get_enemy().copy()
        self.enemy_pos = [self.enemy[0],self.enemy[1]]

        self.navi_state = 0 #0:未启动，1：执行中，2：已到达

        self.killed = False

        return self._get_obs() , self._get_info()

    def _get_obs(self):
        return {
            "agent": np.array(self.agent_pos, dtype=np.int32),
            'enemy': np.array(self.enemy, dtype=np.int32),
            'HP': self.curr_HP,
            "navi_state":self.navi_state
        }
    
    def _get_info(self):
        
        return {
            "agent": self.agent_pos,
            'enemy': self.enemy_pos,
            'HP': self.curr_HP,
            "navi_state":self.navi_state
            
        }



    def step(self, action):
        reward = 0
        action_type = action[0]
        action_dic = [action[1],action[2]]
        terminated = False
        if_hitted = "None"
        if action_type == 0:# use navi 

            if self.navi_state == 0:
                self.navi_state = 1

            reward -= 1

            navi_action, _states = self.navi.predict(self.get_navi_obs(action_dic))

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

            if self.agent_pos[0] == action_dic[0] and self.agent_pos[1] == action_dic[1]:
                self.navi_state = 2
                
                    
        elif (action_type == 1):#attack
            
            if self.navi_state == 1:
                reward -=10
                self.navi_state = 0


            next_x= self.agent_pos[0]
            next_y =self.agent_pos[1]
            dic_x= action_dic[0] #[0-14]
            dic_y= action_dic[1]#[0-14]

            if(dic_x == 7 and dic_y > 7):#up
                next_y-=1
            elif(dic_x > 7 and dic_y > 7):#up_right
                next_y-=1
                next_x+=1
            elif(dic_x == 7 and dic_y < 7):#down
                next_y+=1
            elif(dic_x > 7 and dic_y < 7):#down_right
                next_y+=1
                next_x+=1
            elif(dic_x > 7 and dic_y == 7):#right
                next_x+=1
            elif(dic_x < 7 and dic_y == 7):#left
                next_x-=1
            elif(dic_x < 7 and dic_y > 7):#up_left
                next_x-=1
                next_y-=1
            elif(dic_x < 7 and dic_y < 7):#down_left
                next_x-=1
                next_y+=1
                
            
            if self.enemy_pos[0] == next_x and self.enemy_pos[1] == next_y :
                self.enemy[2] -=2
                reward +=20
                if_hitted = "hitted"
                if self.enemy[2]<=0:
                    reward +=20
                    terminated = True
            else:
                if_hitted = "loss"
                reward -=1.5
        
        observation = self._get_obs()
        info = self._get_info()
        print(action,info,if_hitted)
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
            Hp = random.randint(1, 6)
            if self.get_distance(self.agent_pos,[x,y] )>= 9:
                enemy = [x,y,Hp]
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
    id='attack_Env-v0',
    entry_point='Envs.attack_Env:attack_Env',
    max_episode_steps=100,
)