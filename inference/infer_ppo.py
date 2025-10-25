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
    

    env = gym.make(env_id, render_mode='human',use_lidar=False)

    
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]

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
    
    # buffer = RolloutBuffer(
    #     num_steps=num_steps,
    #     num_envs=num_envs,
    #     state_shape=(state_dim,),
    #     action_shape=(),
    #     device=device
    # )
    
    ob, _ = env.reset()
    # print("obs shape:", obs.shape)
    ob = torch.FloatTensor(ob).to(device)
    
    state_dict = torch.load("../training/flappy_bird_ppo_checkpoints/best_model.pth",weights_only=False)
    agent.load_state_dict(state_dict["model_state_dict"])
    agent.eval()
    rewards_per_episode = []

    for episode in itertools.count():

            state, _ = env.reset()  # Initialize environment. Reset returns (state,info).
            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode

            # Perform actions until episode terminates or reaches max rewards
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
            while(not terminated):

                
                # select best action
                with torch.no_grad():
                    # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
                    # policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3]).
                    # argmax finds the index of the largest element.
                    action,_,_,_ = agent.get_action_and_value(state)

                # Execute action. Truncated and info is not used.
                new_state,reward,terminated,truncated,info = env.step(action.item())

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)

            
if __name__ == "__main__":
    train()
