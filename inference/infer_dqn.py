import sys
import os
import yaml
import torch
import gymnasium as gym
import numpy as np
from datetime import datetime,timedelta
import flappy_bird_gymnasium
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.dqn import DQN
from algorithm.experience_replay import *
from utils import *

def config():
    with open('../hyperparameters.yml', 'r') as file:
        all_hyperparameters = yaml.safe_load(file)
        return all_hyperparameters['flappybird1']

def infer():
    hyperparameters = config()
    env_id = hyperparameters['env_id']
    learning_rate_a = hyperparameters['learning_rate_a']
    discount_factor_g = hyperparameters['discount_factor_g']
    network_sync_rate = hyperparameters['network_sync_rate']
    replay_memory_size = hyperparameters['replay_memory_size']
    mini_batch_size = int(hyperparameters['mini_batch_size'])
    epsilon_init = hyperparameters['epsilon_init']
    epsilon_decay = hyperparameters['epsilon_decay']
    epsilon_min = hyperparameters['epsilon_min']
    stop_on_reward = hyperparameters['stop_on_reward']
    fc1_nodes = hyperparameters['fc1_nodes']
    env_make_params = hyperparameters.get('env_make_params',{})
    enable_double_dqn = hyperparameters['enable_double_dqn']
    enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
    hidden_dim = hyperparameters['hidden_dim']
    training_episodes = hyperparameters['training_episodes']
    # state_dim = config_params['state_dim']
    # action_dim = config_params['action_dim']
    # layer_size = con['layer_size']
    env = gym.make(env_id, render_mode='human',use_lidar=False)
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    agent = DQN(state_dim,action_dim,hidden_dim,enable_dueling_dqn)
    state_dict = torch.load("../best_model.pth",weights_only=False)
    agent.load_state_dict(state_dict["model_state_dict"])
    agent.eval()
    rewards_per_episode = []
    for episode in itertools.count():

            state, _ = env.reset()  
            terminated = False      
            episode_reward = 0.0 
            done = False 
            while(done == False):    
                with torch.no_grad():
                    state = torch.tensor(state, dtype=torch.float, device=device)
                    action = agent(state).argmax(dim=0)
                new_state,reward,terminated,truncated,info = env.step(action.item())
                if terminated or truncated:
                     done = True
                episode_reward += reward
                state = new_state
            rewards_per_episode.append(episode_reward)

            
if __name__ == "__main__":
    infer()
