import sys
import os
import yaml
import torch
import gymnasium as gym
import numpy as np
from datetime import datetime
import flappy_bird_gymnasium  

from gymnasium.vector import AsyncVectorEnv
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

from gymnasium.wrappers import GrayscaleObservation as _GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation as _FrameStack
from gymnasium.wrappers import ResizeObservation as _ResizeObservation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.cnn import PolicyGradient_CNN
from algorithm.rollout_buffer import RolloutBuffer
from utils import save_checkpoint, log_performance, plot_rewards


def config():
    with open("../hyperparameters.yml", "r", encoding="utf-8") as file:
        all_hyperparameters = yaml.safe_load(file)
        return all_hyperparameters["policy_gradient_flappybird"]


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name(0)}")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
    print(f"Using {torch.xpu.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("No cuda or xpu, using cpu")

CPU_DEVICE = torch.device("cpu")


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = int(skip)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None

        for _ in range(max(1, self.skip)):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class FixFlappyStepAPI(gym.Wrapper):
    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 6:
            obs, reward, terminated, score_limit_reached,truncated, info = out
            truncated = bool(truncated or score_limit_reached)
            return obs, reward, terminated, truncated, info
        return out 

class RenderObservation(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(512, 288, 3), dtype=np.uint8)

    def observation(self, obs):
        frame = self.env.render()
        return frame


def make_env(env_id: str, seed: int, idx: int, num_frames: int,frame_skip: int = 1):
    def thunk():
        
        
        env = gym.make(env_id, render_mode="rgb_array", disable_env_checker=True, use_lidar=False)

        env = FixFlappyStepAPI(env)
        # raw = env.step(env.action_space.sample())
        # raw = env.reset()
        # raw = env.step(env.action_space.sample())
        # print("================= raw step len =", len(raw), "keys(info) =", list(raw[-1].keys()) if isinstance(raw, tuple) else None)
        
        if int(frame_skip) > 1:
            env = ActionRepeat(env, skip=int(frame_skip))

        env = RenderObservation(env)
        # env = gym.wrappers.RecordEpisodeStatistics(env)

        env = _ResizeObservation(env, (84, 84))
        try:
            env = _GrayscaleObservation(env, keep_dim=False)
        except TypeError:
            env = _GrayscaleObservation(env)

        env = _FrameStack(env, num_frames)
        env.action_space.seed(seed + idx)
        return env
    return thunk



def _obs_to_tensor_for_cnn(obs, device: torch.device) -> torch.Tensor:
    obs = np.array(obs)
    x = torch.as_tensor(obs, device=device)

    if x.ndim == 5 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if x.ndim == 5 and x.shape[-2] == 1:
        x = x.squeeze(-2)

    if x.ndim == 4:
        if x.shape[-1] in (1, 2, 3, 4) and x.shape[1] not in (1, 2, 3, 4):
            x = x.permute(0, 3, 1, 2).contiguous()

    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    else:
        x = x.float()

    return x


def train_cnn_policy_gradient():
    cfg = config()

    env_id = cfg["env_id"]
    num_envs = int(cfg["num_envs"])
    num_steps = int(cfg["num_steps"])
    total_epochs = int(cfg["total_epochs"])

    gamma = float(cfg["gamma"])
    gae_lambda = float(cfg["gae_lambda"])

    vf_coef = float(cfg["vf_coef"])
    ent_coef = float(cfg["ent_coef"])
    lr = float(cfg["lr"])
    max_grad_norm = float(cfg["max_grad_norm"])
    num_minibatches = int(cfg["num_minibatches"])
    anneal_lr = bool(cfg.get("anneal_lr", True)) 

    hidden_dim = int(cfg["hidden_dim"])
    use_baseline = bool(cfg.get("use_baseline", True))

    num_frames = int(cfg["frame_stack"])
    seed = int(cfg.get("seed", 42))
    frame_skip = int(cfg.get("frame_skip", 1))

    np.random.seed(seed)
    torch.manual_seed(seed)

    envs = AsyncVectorEnv([make_env(env_id, seed, i, num_frames,frame_skip) for i in range(num_envs)])
    action_dim = envs.single_action_space.n

    obs, _ = envs.reset(seed=seed)
    
    obs_cpu0 = _obs_to_tensor_for_cnn(obs, CPU_DEVICE)
    print(obs.shape)
    cnn_state_shape = tuple(obs_cpu0.shape[1:]) 
    agent = PolicyGradient_CNN(
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        lr=lr,
        max_grad_norm=max_grad_norm,
        obs_shape=cnn_state_shape,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        use_baseline=use_baseline,
        cfg=cfg,
    ).to(DEVICE)

    
    agent.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    agent.train()

    #Warm-up
    with torch.no_grad():
        dummy = torch.zeros((num_envs, *cnn_state_shape), device=DEVICE, dtype=torch.float32)
        agent.get_action_and_value(dummy)
        if hasattr(torch, "xpu") and DEVICE.type == "xpu":
            torch.xpu.synchronize()
    print("Warm-up done.", flush=True)

    optimizer = agent.optimizer

    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        state_shape=cnn_state_shape,
        action_shape=1,
        device=CPU_DEVICE,
    )

    best_avg_reward = -float("inf")
    episode_rewards = []
    episode_lengths = []
    current_episode_rewards = np.zeros(num_envs, dtype=np.float32)
    current_episode_lengths = np.zeros(num_envs, dtype=np.int32)

    print(f"Bat dau train ActorCritic(CNN) voi game: {env_id}")
    print(f"Total epochs: {total_epochs}, Steps per epoch: {num_steps}, Num envs: {num_envs}")

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(total_epochs):
        if anneal_lr:
            frac = 1.0 - epoch/ total_epochs
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(num_steps):
            obs_cpu = _obs_to_tensor_for_cnn(obs, CPU_DEVICE)

            obs_dev = obs_cpu.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                actions_dev, logprobs_dev, _, values_dev = agent.get_action_and_value(obs_dev)

            actions_np = actions_dev.detach().cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = envs.step(actions_np)
            dones_np = np.logical_or(terminated, truncated)
            
            dones_buffer = terminated

            actions_cpu = actions_dev.detach().cpu().long().view(-1)
            logprobs_cpu = logprobs_dev.detach().cpu().view(-1)
            values_cpu = values_dev.detach().cpu().view(-1)
            rewards_cpu = torch.as_tensor(rewards, dtype=torch.float32, device=CPU_DEVICE).view(-1)
            dones_cpu = torch.as_tensor(dones_buffer, dtype=torch.float32, device=CPU_DEVICE).view(-1)

            buffer.add(obs_cpu, actions_cpu, logprobs_cpu, rewards_cpu, dones_cpu, values_cpu)

            current_episode_rewards += rewards.astype(np.float32)
            current_episode_lengths += 1
            
            for i in range(num_envs):
                if dones_np[i]:
                    episode_rewards.append(float(current_episode_rewards[i]))
                    episode_lengths.append(int(current_episode_lengths[i]))
                    current_episode_rewards[i] = 0.0
                    current_episode_lengths[i] = 0

            obs = next_obs

        with torch.no_grad():
            last_obs_cpu = _obs_to_tensor_for_cnn(obs, CPU_DEVICE)
            last_obs_dev = last_obs_cpu.to(DEVICE, non_blocking=True)
            _, _, _, last_values_dev = agent.get_action_and_value(last_obs_dev)

            last_values = last_values_dev.detach().cpu().flatten()
            last_dones = dones_cpu  #cpu tensor
            buffer.compute_returns_and_advantages(last_values, last_dones, gamma, gae_lambda)

        loss = agent.update(buffer)

        if episode_rewards:
            avg_reward = float(np.mean(episode_rewards[-100:]))
            avg_length = float(np.mean(episode_lengths[-100:]))
        else:
            avg_reward = 0.0
            avg_length = 0.0

        log_performance(
            epoch=epoch,
            avg_reward=avg_reward,
            loss=loss,
            path=f"flappy_bird_cnn_policy_gradient/{time_str}/train_performance_log.csv",
        )

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{total_epochs}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Length (last 100): {avg_length:.2f}")
            # print(f"  Episodes completed: {len(episode_rewards)}")

        if epoch % 100 == 99:
            # best_avg_reward = avg_reward
            save_checkpoint(
                model=agent,
                optimizer=optimizer,
                epoch=epoch,
                avg_reward=avg_reward,
                path="flappy_bird_cnn_policy_gradient_checkpoints",
            )
            print(f"  New model saved! Avg reward: {avg_reward:.2f}")

        buffer.clear()

    save_checkpoint(
        model=agent,
        optimizer=optimizer,
        epoch=total_epochs,
        avg_reward=best_avg_reward if episode_rewards else 0.0,
        path="flappy_bird_cnn_policy_gradient_checkpoints",
    )

    plot_rewards(
        log_dir="flappy_bird_cnn_policy_gradient",
        window=10,
        save_path=f"flappy_bird_cnn_policy_gradient/final_training_results_{time_str}.png",
    )

    envs.close()
    print("Training finished...")
    print(f"Best average reward: {best_avg_reward:.2f}")
    # print(f"Total episodes completed: {len(episode_rewards)}")


if __name__ == "__main__":
    train_cnn_policy_gradient()