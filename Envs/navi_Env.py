from typing import List
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import random
from gymnasium.envs.registration import register
import math



class navi_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

     
     
    def __init__(self,render_mode = "human",width = 15,height = 15):
        super().__init__()
        self.width = width
        self.height = height

        self.agent_pos = self.random_Agent_pos().copy()
        self.goal=self.get_goal
        self.arrive =False

        self.curr_dis = 10000

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0,15,(2 ,),np.int32),
            "goal": spaces.Box(0,15,(2 ,),np.int32),
            })

        
            
        self.action_space = spaces.Discrete(4)

    def reset(self,seed=None, options=None):
        
        self.agent_pos = self.random_Agent_pos().copy()
        self.goal=self.get_goal
        self.arrive =False

        return self._get_obs() , self._get_info()

    def _get_obs(self):
        return {
            "agent": np.array(self.agent_pos, dtype=np.int32),
            "goal": np.array(self.goal, dtype=np.int32),
        }
    
    def _get_info(self):
        
        return {
            "agent": self.agent_pos,
            'enemy': self.goal,
            "arrive":self.arrive,
            
        }
    
    def step(self, action):

        reward = -1
        terminated = False
        next_x= self.agent_pos[0]
        next_y =self.agent_pos[1]
        if(action == 0):#up
            next_y-=1
        elif(action == 1):#down
            next_y+=1
        elif(action == 2):#right
            next_x+=1
        elif(action == 3):#left
            next_x-=1
        
        if(next_x == self.goal[0]and next_y == self.goal[1]):
            reward += 50
            self.arrive = True
            terminated = True
        elif(next_x < 0 or next_x >=self.width or next_y < 0 or next_y >=self.height):
            reward -= 5
        else:
            self._update_agent_position(next_x,next_y)

                    

        
        
        observation = self._get_obs()
        info = self._get_info()
        print(action,info)
        return observation, reward, terminated, False, info

    

    def _update_agent_position(self, next_x,next_y):

        self.agent_pos = (next_x, next_y)
    

    def random_Agent_pos(self):
        x = random.randint(0, 14)
        y = random.randint(0, 14)
        return [x,y]



    def get_distance(self,pos1,pos2):
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return distance

    def get_goal(self):
        goal = [-1,-1]
        while True:
            x = random.randint(0, 14)
            y = random.randint(0, 14)
            if self.get_distance(self.agent_pos,[x,y]) >= 6:
                goal = [x,y]
                break
        return goal
        
register(
    id='navi_Env-v0',
    entry_point='Envs.navi_Env:navi_Env',
    max_episode_steps=100,
)
        