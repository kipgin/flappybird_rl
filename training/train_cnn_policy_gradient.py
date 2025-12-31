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


class RenderObservation(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(512, 288, 3), dtype=np.uint8)

    def observation(self, obs):
        frame = self.env.render()
        return frame


def make_env(env_id: str, seed: int, idx: int, num_frames: int):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array", use_lidar=False)
        env = RenderObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

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
    debug = bool(cfg.get("debug", False))

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

    hidden_dim = int(cfg["hidden_dim"])
    use_baseline = bool(cfg.get("use_baseline", True))

    num_frames = int(cfg["frame_stack"])
    seed = int(cfg.get("seed", 42))

    np.random.seed(seed)
    torch.manual_seed(seed)

    envs = AsyncVectorEnv([make_env(env_id, seed, i, num_frames) for i in range(num_envs)])
    action_dim = envs.single_action_space.n

    obs, _ = envs.reset(seed=seed)
    obs_cpu0 = _obs_to_tensor_for_cnn(obs, CPU_DEVICE)
    cnn_state_shape = tuple(obs_cpu0.shape[1:]) 
    agent_learner = PolicyGradient_CNN(
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

    agent_actor = PolicyGradient_CNN(
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        lr=lr,
        max_grad_norm=max_grad_norm,
        obs_shape=cnn_state_shape,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        use_baseline=use_baseline,
        cfg=cfg,
    ).to(CPU_DEVICE)

    agent_actor.load_state_dict(agent_learner.state_dict())
    agent_actor.eval()
    agent_learner.train()

    # Warm-up (materialize LazyLinear + compile kernels early)
    with torch.no_grad():
        dummy = torch.zeros((num_envs, *cnn_state_shape), device=DEVICE, dtype=torch.float32)
        agent_learner.get_action_and_value(dummy)
        if debug and hasattr(torch, "xpu") and DEVICE.type == "xpu":
            torch.xpu.synchronize()
    print("Warm-up learner done.", flush=True)

    optimizer = agent_learner.optimizer

    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        state_shape=cnn_state_shape,
        action_shape=1,
        device=DEVICE,
    )

    best_avg_reward = -float("inf")
    episode_rewards = []
    episode_lengths = []
    current_episode_rewards = np.zeros(num_envs, dtype=np.float32)
    current_episode_lengths = np.zeros(num_envs, dtype=np.int32)

    print(f"Bat dau train PolicyGradient(CNN) voi game: {env_id}")
    print(f"Total epochs: {total_epochs}, Steps per epoch: {num_steps}, Num envs: {num_envs}")

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(total_epochs):
        for step in range(num_steps):
            obs_cpu = _obs_to_tensor_for_cnn(obs, CPU_DEVICE)

            with torch.no_grad():
                actions_cpu, logprobs_cpu, _, values_cpu = agent_actor.get_action_and_value(obs_cpu)

            next_obs, rewards, terminated, truncated, infos = envs.step(actions_cpu.cpu().numpy())
            dones_np = np.logical_or(terminated, truncated)

            obs_dev = obs_cpu.to(DEVICE, non_blocking=True)
            actions_dev = actions_cpu.to(DEVICE, non_blocking=True).long().view(-1)
            logprobs_dev = logprobs_cpu.to(DEVICE, non_blocking=True).view(-1)
            rewards_dev = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE).view(-1)
            dones_dev = torch.as_tensor(dones_np, dtype=torch.float32, device=DEVICE).view(-1)
            values_dev = values_cpu.to(DEVICE, non_blocking=True).view(-1)

            buffer.add(obs_dev, actions_dev, logprobs_dev, rewards_dev, dones_dev, values_dev)

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
            _, _, _, last_values_cpu = agent_actor.get_action_and_value(last_obs_cpu)
            last_values = last_values_cpu.flatten().to(DEVICE)
            last_dones = dones_dev
            buffer.compute_returns_and_advantages(last_values, last_dones, gamma, gae_lambda)

        loss = agent_learner.update(buffer)

       
        agent_actor.load_state_dict(agent_learner.state_dict())
        agent_actor.eval()

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
            print(f"  Episodes completed: {len(episode_rewards)}")

        if episode_rewards and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            save_checkpoint(
                model=agent_learner,
                optimizer=optimizer,
                epoch=epoch,
                avg_reward=avg_reward,
                path="flappy_bird_cnn_policy_gradient_checkpoints",
            )
            print(f"  New best model saved! Avg reward: {best_avg_reward:.2f}")

        buffer.clear()

    save_checkpoint(
        model=agent_learner,
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
    print(f"Total episodes completed: {len(episode_rewards)}")


if __name__ == "__main__":
    train_cnn_policy_gradient()