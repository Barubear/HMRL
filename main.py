
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

import train


#from Envs.AOE_Env import AOE_Env
#from Envs.shoot_Env import shoot_Env

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


def train_aoe():
    save_path = 'trained_modules/AOE_Module/normal_best'
    log_path = 'logs/AOE_log'
    env = make_vec_env("AOE_Env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,2000000)

def train_shoot():
    save_path = 'trained_modules/shoot_Module/normal_best02'
    log_path = 'logs/shoot_log'
    env = make_vec_env("shoot_Env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,1000000)

train_shoot()