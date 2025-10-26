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

from algorithm.policy_gradient import PolicyGradient
from algorithm.rollout_buffer import RolloutBuffer
from utils import *

def config():
    with open('../hyperparameters.yml', 'r') as file:
        all_hyperparameters = yaml.safe_load(file)
        return all_hyperparameters['policy_gradient_flappybird']

def infer():

    config_params = config()
    env_id = config_params['env_id']
    num_envs = config_params['num_envs']
    num_steps = config_params['num_steps']
    total_epochs = config_params['total_epochs']
    gamma = config_params['gamma']
    gae_lambda = config_params['gae_lambda']
    device = config_params['device']
    # clip_coef = config_params['clip_coef']
    vf_coef = config_params['vf_coef']
    ent_coef = config_params['ent_coef']
    lr = config_params['lr']
    max_grad_norm = config_params['max_grad_norm']
    # update_epochs = config_params['update_epochs']
    # num_minibatches = config_params['num_minibatches']
    # state_dim = config_params['state_dim']
    # action_dim = config_params['action_dim']
    hidden_dim = config_params['hidden_dim']

    env = gym.make(env_id, render_mode='human',use_lidar=False)

    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]

    agent = PolicyGradient(
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        lr=lr,
        max_grad_norm=max_grad_norm,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        use_baseline=True
    ).to(device)
    
    state_dict = torch.load("../training/flappy_bird_policy_gradient_checkpoints/best_model.pth",weights_only=False)
    agent.load_state_dict(state_dict["model_state_dict"])
    agent.eval()
    rewards_per_episode = []
    for episode in itertools.count():
        state, _ = env.reset()  
        state = torch.tensor(state, dtype=torch.float, device=device) 
        stop = False
        episode_reward = 0.0   
        while(stop == False):
            with torch.no_grad():
                action,_,_,_ = agent.get_action_and_value(state)
            new_state,reward,terminated,truncated,info = env.step(action.item())
            if terminated or truncated:
                    stop = True
            episode_reward += reward
            new_state = torch.tensor(new_state, dtype=torch.float, device=device)
            reward = torch.tensor(reward, dtype=torch.float, device=device)
            state = new_state
        rewards_per_episode.append(episode_reward)
        if episode % 100 == 99:
            latest_rewards =  rewards_per_episode[-100:]
            avg_reward = np.mean(latest_rewards)
            print(f"---------\nThe average reward is: {avg_reward}")
if __name__ == "__main__":
    infer()
    
