import sys
import os
import yaml
import torch
import gymnasium as gym
import numpy as np
from datetime import datetime,timedelta
import flappy_bird_gymnasium
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.policy_gradient import *
from algorithm.rollout_buffer import RolloutBuffer
from utils import save_checkpoint, log_performance, plot_rewards
from gymnasium.vector import AsyncVectorEnv

def config():
    with open('../hyperparameters.yml', 'r') as file:
        all_hyperparameters = yaml.safe_load(file)
        return all_hyperparameters['policy_gradient_flappybird']


device = ''

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using {torch.cuda.get_device_name(0)}")

elif torch.xpu.is_available():
    device = torch.device("xpu")
    print(f"Using {torch.xpu.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No cuda or xpu, using cpu")



def train():

    config_params = config()
    env_id = config_params['env_id']
    num_envs = config_params['num_envs']
    num_steps = config_params['num_steps']
    total_epochs = config_params['total_epochs']
    gamma = config_params['gamma']
    gae_lambda = config_params['gae_lambda']
    # device = config_params['device']
    # clip_coef = config_params['clip_coef']
    vf_coef = config_params['vf_coef']
    ent_coef = config_params['ent_coef']
    lr = config_params['lr']
    max_grad_norm = config_params['max_grad_norm']
    update_epochs = config_params['update_epochs']
    # num_minibatches = config_params['num_minibatches']
    # state_dim = config_params['state_dim']
    # action_dim = config_params['action_dim']
    hidden_dim = config_params['hidden_dim']
    

    # envs = gym.vector.make(env_id, num_envs=num_envs,use_lidar=False)
    # envs = gym.make_vec(env_id,num_envs=num_envs,use_lidar = False)
    envs  = AsyncVectorEnv([lambda : gym.make(env_id,use_lidar = False) for _ in range(num_envs)])
    action_dim = envs.single_action_space.n
    state_dim = envs.single_observation_space.shape[0]

    print(action_dim,state_dim)

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
    
    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        state_shape=state_dim,
        action_shape=action_dim,
        device=device
    )
    
    # obs, _ = envs.reset()
    # print("obs shape:", obs.shape)
    # obs = torch.FloatTensor(obs).to(device)
    
    best_avg_reward = -float('inf')
    episode_rewards = []
    episode_lengths = []
    current_episode_rewards = np.zeros(num_envs)
    current_episode_lengths = np.zeros(num_envs)
    
    print(f"Bat dau train Policy Gradient voi game: {env_id}")
    print(f"Total epochs: {total_epochs}, Steps per epoch: {num_steps}")
    
    for epoch in range(total_epochs):
        obs, _ = envs.reset()
        # print("obs shape:", obs.shape)
        obs = torch.FloatTensor(obs).to(device)
        obs = torch.tensor(obs,dtype=torch.float).to(device)
        for step in range(num_steps):
            with torch.no_grad():
                actions, logprobs,entropy, values = agent.get_action_and_value(obs)
            next_obs, rewards, dones, truncateds, infos = envs.step(actions.cpu().numpy())
            # actions = torch.o
            # print(actions.shape)
            next_obs = torch.tensor(next_obs,dtype=torch.float32).to(device)
            rewards = torch.tensor(rewards,dtype = torch.float32).to(device)
            dones = torch.tensor(dones,dtype=torch.int64).to(device)
            buffer.add(obs, actions, logprobs, rewards, dones, values)
            
            current_episode_rewards += rewards.cpu().numpy()
            current_episode_lengths += 1
            
            for i in range(num_envs):
                if dones[i] or truncateds[i]:
                    episode_rewards.append(current_episode_rewards[i])
                    episode_lengths.append(current_episode_lengths[i])
                    current_episode_rewards[i] = 0
                    current_episode_lengths[i] = 0
            
            obs = next_obs
        
        with torch.no_grad():
            last_value = agent.critic(obs).flatten()
            last_done = dones
            buffer.compute_returns_and_advantages(last_value, last_done, gamma, gae_lambda)
        
        loss = agent.update(buffer)
        
        # Calculate metrics
        if episode_rewards:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
        else:
            avg_reward = 0.0
            avg_length = 0.0
            
        log_performance(
            epoch=epoch,
            avg_reward=avg_reward,
            loss=loss, 
            path="flappy_bird_policy_gradient/train_performance_log.csv"
        )
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{total_epochs}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Length (last 100): {avg_length:.2f}")
            print(f"  Episodes completed: {len(episode_rewards)}")
        
        if epoch % 100 == 99:
            # best_avg_reward = avg_reward
            save_checkpoint(
                model=agent,
                optimizer=agent.optimizer,
                epoch=epoch,
                avg_reward=avg_reward,
                path="flappy_bird_policy_gradient_checkpoints"
            )
            print(f"  New model saved! Avg reward: {avg_reward:.2f}")
        buffer.clear()
    
    save_checkpoint(
        model=agent,
        optimizer=agent.optimizer,
        epoch=total_epochs,
        avg_reward=avg_reward if episode_rewards else 0,
        path="flappy_bird_policy_gradient_checkpoints"
    )

    print("Training completed...")
    plot_rewards(
        log_dir="flappy_bird_policy_gradient",
        window=10,
        save_path="flappy_bird_policy_gradient/final_training_results.png"
    )
    
    envs.close()
    
    print(f"Training finished...")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"Total episodes completed: {len(episode_rewards)}")

if __name__ == "__main__":
    train()
