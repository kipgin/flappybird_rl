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

from algorithm.ppo import PPO
from algorithm.rollout_buffer import RolloutBuffer
from utils import save_checkpoint, log_performance, plot_rewards

def config():
    with open('../hyperparameters.yml', 'r') as file:
        all_hyperparameters = yaml.safe_load(file)
        return all_hyperparameters['ppo_flappybird']

def train():

    config_params = config()
    env_id = config_params['env_id']
    num_envs = config_params['num_envs']
    num_steps = config_params['num_steps']
    total_epochs = config_params['total_epochs']
    gamma = config_params['gamma']
    gae_lambda = config_params['gae_lambda']
    device = config_params['device']
    clip_coef = config_params['clip_coef']
    vf_coef = config_params['vf_coef']
    ent_coef = config_params['ent_coef']
    lr = config_params['lr']
    max_grad_norm = config_params['max_grad_norm']
    update_epochs = config_params['update_epochs']
    num_minibatches = config_params['num_minibatches']
    # state_dim = config_params['state_dim']
    # action_dim = config_params['action_dim']
    layer_size = config_params['layer_size']
    

    # envs = gym.vector.make(env_id, num_envs=num_envs,use_lidar=False)
    envs = gym.make_vec(env_id,num_envs=num_envs,use_lidar = False)
    action_dim = envs.single_action_space.n
    state_dim = envs.single_observation_space.shape[0]

    agent = PPO(
        clip_coef=clip_coef,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        lr=lr,
        max_grad_norm=max_grad_norm,
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        state_dim=state_dim,
        action_dim=action_dim,
        layer_size=layer_size
    ).to(device)
    
    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        state_shape=state_dim,
        action_shape=action_dim,
        device=device
    )
    
    obs, _ = envs.reset()
    print("obs shape:", obs.shape)
    obs = torch.FloatTensor(obs).to(device)
    
    best_avg_reward = -float('inf')
    episode_rewards = []
    episode_lengths = []
    current_episode_rewards = np.zeros(num_envs)
    current_episode_lengths = np.zeros(num_envs)
    
    print(f"Bat dau train PPO voi game: {env_id}")
    print(f"Total epochs: {total_epochs}, Steps per epoch: {num_steps}")
    
    for epoch in range(total_epochs):
        for step in range(num_steps):
            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(obs)
            # values = values.flatten()
            next_obs, rewards, dones, truncated, info = envs.step(actions.cpu().numpy())
    
            next_obs = torch.tensor(next_obs,dtype=torch.float32).to(device)
            rewards = torch.tensor(rewards,dtype=torch.float32).to(device)
            dones = torch.tensor(dones,dtype = torch.int64).to(device)
            
            buffer.add(obs, actions, logprobs, rewards, dones, values)
            
            current_episode_rewards += rewards.cpu().numpy()
            current_episode_lengths += 1
            
            for i in range(num_envs):
                if dones[i] or truncated[i]:
                    episode_rewards.append(current_episode_rewards[i])
                    episode_lengths.append(current_episode_lengths[i])
                    current_episode_rewards[i] = 0
                    current_episode_lengths[i] = 0
            
            obs = next_obs
        
        with torch.no_grad():
            last_values = agent.get_value(obs).flatten()
            last_dones = dones
            buffer.compute_returns_and_advantages(last_values, last_dones, gamma, gae_lambda)
        
        loss = agent.update(buffer)
        
        # Calculate metrics
        if episode_rewards:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
        else:
            avg_reward = 0.0
            avg_length = 0.0
        # print("Vua train xong 1 lan")
        log_performance(
            epoch=epoch,
            avg_reward=avg_reward,
            loss=loss, 
            path="flappy_bird_ppo/train_performance_log.csv"
        )
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{total_epochs}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Length (last 100): {avg_length:.2f}")
            print(f"  Episodes completed: {len(episode_rewards)}")
        
        if episode_rewards and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            save_checkpoint(
                model=agent,
                optimizer=agent.optimizer,
                epoch=epoch,
                avg_reward=avg_reward,
                path="flappy_bird_ppo_checkpoints"
            )
            print(f"  New best model saved! Avg reward: {avg_reward:.2f}")
        buffer.clear()
    
    save_checkpoint(
        model=agent,
        optimizer=agent.optimizer,
        epoch=total_epochs,
        avg_reward=avg_reward if episode_rewards else 0,
        path="flappy_bird_ppo_checkpoints"
    )

    print("Training completed...")
    plot_rewards(
        log_dir="flappy_bird_ppo",
        window=10,
        save_path="flappy_bird_ppo/final_training_results.png"
    )
    
    envs.close()
    
    print(f"Training finished...")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"Total episodes completed: {len(episode_rewards)}")

if __name__ == "__main__":
    train()
