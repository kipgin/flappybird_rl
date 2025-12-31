import sys
import os
import yaml
from torch import nn
import torch
import gymnasium as gym
# import numpy as np
from datetime import datetime,timedelta
import flappy_bird_gymnasium
import numpy as np
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import random


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.dqn import DQN
from algorithm.experience_replay import ReplayBuffer
from utils import save_checkpoint, log_performance, plot_rewards
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


def config():
    with open('../hyperparameters.yml', 'r') as file:
        all_hyperparameters = yaml.safe_load(file)
        return all_hyperparameters['flappybird1']

def train():

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

    env = gym.make(env_id,use_lidar=False)
    
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]

    agent = DQN(state_dim, action_dim, hidden_dim, enable_dueling_dqn=enable_dueling_dqn).to(device)
    target_agent = DQN(state_dim, action_dim, hidden_dim, enable_dueling_dqn=enable_dueling_dqn).to(device)
    target_agent.load_state_dict(agent.state_dict())

    buffer = ReplayBuffer(state_dim,action_dim,replay_memory_size,mini_batch_size)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate_a)
    # ob, _ = env.reset()
    # print("obs shape:", ob.shape)
    # ob = torch.FloatTensor(ob).to(device)
    
    losses = []
    
    print(f"Bat dau train DQN voi game: {env_id}")
    print(f"Total epochs: {training_episodes}")
    
    # epsilons = [epsilon_init]
    # last_element = epsilon_init
    # for x in range(training_episodes):
    #     if(x == 0):
    #         continue
    #     item = max(last_element*epsilon_decay,epsilon_min)
    #     epsilons.append(item)
    #     last_element = item
    
    # epsilons = torch.tensor(epsilons,dtype=torch.float, device=device)
    # print(epsilons)
    # return 0

    epsilon = epsilon_init

    episode_rewards = []
    episode_lengths = []
    loss_fn = nn.MSELoss()

    cnt = 0
    for epoch in range(training_episodes):
        state,_ = env.reset()
        state = torch.tensor(state,dtype=torch.float32, device=device)
        done = False
        episode_reward = 0.0
        episode_length = 0

        losses_in_episode =[]
        done = 0
        while(done == 0):
            # optimizer = agent.optimizer
            # if torch.rand(1).item() < 1 :
            if torch.rand(1).item() < epsilon :
                action = env.action_space.sample()
                action = torch.tensor(action,dtype = torch.int64,device= device)
            

            else:
                with torch.no_grad():
                    # action = agent(state.unsqueeze(dim=0)).squeeze().argmax()
                    action = agent(state).argmax()

            # print(action)
            # print(type(action))
            # return 0

            next_state, reward, terminated, truncated, info = env.step(action.item())

            if terminated or truncated:
                done = 1
            else :
                done = 0

            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            reward = torch.tensor(reward, dtype=torch.float32, device=device)
            done = torch.tensor(done, dtype=torch.int64, device=device)
            action = torch.tensor(action, dtype=torch.int64, device=device)

            # epsilon = max(epsilon*epsilon_decay,epsilon_min)

            transition = (state, action, reward, next_state, done)
            
            buffer.add(transition)
            
            episode_reward += reward.cpu().numpy()
            episode_length += 1
            
            if buffer.real_size >= mini_batch_size :
                mini_batch = buffer.sample()
                states,actions,rewards,next_states,dones = mini_batch
                target_q = None
                with torch.no_grad():
                    if enable_double_dqn:
                        best_actions_from_policy = agent(next_states).argmax(dim=1)
                        target_q = rewards + (1-dones) * discount_factor_g * \
                                        target_agent(next_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
                    else:
                        target_q = rewards + (1-dones) * discount_factor_g * target_agent(next_states).max(dim=1)[0]

                # print(states)
                current_q = agent(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
                # print(current_q)
                # return 0
                loss = loss_fn(current_q, target_q)
                losses_in_episode.append(loss.item())
                optimizer.zero_grad()  
                loss.backward()            
                optimizer.step()

                cnt += 1
                if cnt >= network_sync_rate:
                    # print(50*"=" + "UPDATING.....")
                    target_agent.load_state_dict(agent.state_dict()) 
                    cnt = 0
                    # epsilon = max(epsilon*epsilon_decay,epsilon_min)
            
            state = next_state
  
        avg_loss = np.mean(losses_in_episode)
        losses.append(avg_loss)
        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)

        epsilon = max(epsilon*epsilon_decay,epsilon_min)

        log_performance(
            epoch=epoch,
            avg_reward=episode_rewards[epoch],
            loss=losses[epoch], 
            path="flappy_bird_dqn/train_performance_log.csv"
        )
        
        if epoch % 100 == 99:
            save_checkpoint(
                model=agent,
                optimizer = optimizer,
                epoch = epoch,
                avg_reward= episode_rewards[epoch],
                path="flappy_bird_dqn_checkpoints"
            )
            print(f"  New model saved at epoch: {epoch}")
    
    # save_checkpoint(
    #     model=agent,
    #     optimizer = optimizer,
    #     epoch = epoch,
    #     avg_reward= episode_rewards[],
    #     path="flappy_bird_dqn_checkpoints"
    # )

    print("Training completed...")
    plot_rewards(
        log_dir="flappy_bird_dqn",
        window=10,
        save_path="flappy_bird_dqn/final_training_results.png"
    )
    
    env.close()
    
    best_avg_reward = np.max(episode_rewards)

    print(f"Training finished...")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"Total episodes completed: {len(episode_rewards)}")

if __name__ == "__main__":
    train()
