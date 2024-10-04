
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

import train

from Envs.navi_Env import navi_Env
#from Envs.AOE_Env import AOE_Env
from Envs.attack_Env import attack_Env

def Moduletrain(save_path,log_path,env,times = 2000000,testonly =False,re_tarin_model =None):
    model = None
    if re_tarin_model == None:
        model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        learning_rate=1e-4,  # 学习率
        gamma=0.995,  # 折扣因子
        gae_lambda=0.95,  # GAE λ
        clip_range=0.2,  # 剪辑范围
        ent_coef=0.15,  # 熵系数
        batch_size=512,  # 批大小
        n_steps=256,  # 步数
        n_epochs=16,  # 训练次数
        policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1),  # LSTM 设置
        verbose=1,
        )
    else :
        model = re_tarin_model
        model.set_env(env)
    
    train.train(model,env,times,save_path,log_path,100,testonly)


def train_navi():
    save_path = 'trained_modules/navie/normal_best'
    log_path = 'logs/navi_log'
    env = make_vec_env("navi_Env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,1000000,)

def train_attack():
    save_path = 'trained_modules/attack/normal_best'
    log_path = 'logs/attack_log'
    env = make_vec_env("attack_Env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,3000000,)


train_attack()