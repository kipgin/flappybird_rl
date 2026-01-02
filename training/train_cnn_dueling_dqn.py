import sys
import os
import yaml
import torch
import gymnasium as gym
import numpy as np
from datetime import datetime
import flappy_bird_gymnasium  

from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

from gymnasium.wrappers import GrayscaleObservation as _GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation as _FrameStack
from gymnasium.wrappers import ResizeObservation as _ResizeObservation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.cnn import DQN_CNN
from algorithm.experience_replay import ReplayBufferCNN
from utils import save_checkpoint, log_performance, plot_rewards


def config():
    with open("../hyperparameters.yml", "r", encoding="utf-8") as file:
        all_hyperparameters = yaml.safe_load(file)
        return all_hyperparameters["flappybird1"]


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name(0)}")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
    print(f"Using {torch.xpu.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("No cuda or xpu, using cpu")


class RenderObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(512, 288, 3), dtype=np.uint8)

    def observation(self, obs):
        frame = self.env.render()
        return frame


def make_env(env_id: str, env_make_params: dict, num_frames: int, seed: int):
    env_make_params = dict(env_make_params or {})
    env_make_params.setdefault("use_lidar", False)

    env = gym.make(env_id, render_mode="rgb_array", **env_make_params)
    env = RenderObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    env = _ResizeObservation(env, (84, 84))
    try:
        env = _GrayscaleObservation(env, keep_dim=False)
    except TypeError:
        env = _GrayscaleObservation(env)

    env = _FrameStack(env, num_frames)
    env.action_space.seed(seed)
    return env


def _obs_to_tensor_single(obs, device: torch.device) -> torch.Tensor:
    arr = np.array(obs)
    x = torch.as_tensor(arr, device="cpu")
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4) and x.shape[0] not in (1, 3, 4):
        x = x.permute(2, 0, 1).contiguous()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    else:
        x = x.float()
    return x.to(device, non_blocking=True)

def train_cnn_dueling_dqn():
    hp = config()
    env_id = hp["env_id"]
    seed = int(hp.get("seed", 42))

    replay_memory_size = int(hp["replay_memory_size"])
    mini_batch_size = int(hp["mini_batch_size"])

    epsilon_init = float(hp["epsilon_init"])
    epsilon_decay = float(hp["epsilon_decay"])
    epsilon_min = float(hp["epsilon_min"])

    learning_rate_a = float(hp["learning_rate_a"])
    discount_factor_g = float(hp["discount_factor_g"])
    network_sync_rate = int(hp["network_sync_rate"])

    enable_double_dqn = bool(hp.get("enable_double_dqn", False))
    enable_dueling_dqn = bool(hp.get("enable_dueling_dqn", False))

    hidden_dim = int(hp.get("hidden_dim", 128))
    training_episodes = int(hp.get("training_episodes", 1_000_000))

    num_frames = int(hp.get("frame_stack", 4))
    env_make_params = hp.get("env_make_params", {})

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(env_id, env_make_params, num_frames=num_frames, seed=seed)

    action_dim = env.action_space.n
    obs0, _ = env.reset(seed=seed)
    obs0_arr = np.array(obs0)
    obs_shape = tuple(obs0_arr.shape)

    agent = DQN_CNN(
        obs_shape=obs_shape,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        enable_dueling_dqn=enable_dueling_dqn,
        cfg=hp,
    ).to(DEVICE)

    target_agent = DQN_CNN(
        obs_shape=obs_shape,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        enable_dueling_dqn=enable_dueling_dqn,
        cfg=hp,
    ).to(DEVICE)
    target_agent.load_state_dict(agent.state_dict())
    target_agent.eval()

    # Warm-up --> trick dung cho dong intel graphic
    with torch.no_grad():
        dummy = torch.zeros((1, *obs_shape), device=DEVICE, dtype=torch.float32)
        _ = agent(dummy)
        if hasattr(torch, "xpu") and DEVICE.type == "xpu":
            torch.xpu.synchronize()
    print("Warm-up done.", flush=True)

    buffer = ReplayBufferCNN(
        state_shape=obs_shape,
        action_size=action_dim,
        buffer_size=replay_memory_size,
        mini_batch_size=mini_batch_size,
        device=DEVICE,  
    )


    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate_a)
    loss_fn = torch.nn.MSELoss()

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_avg_reward = -float("inf")
    episode_rewards = []
    episode_lengths = []

    epsilon = epsilon_init
    sync_counter = 0

    print(f"Bat dau train CNN+DQN dueling voi game: {env_id}")
    print(f"Episodes: {training_episodes}, Replay: {replay_memory_size}, Batch: {mini_batch_size}, Device: {DEVICE}")

    for epoch in range(training_episodes):
        state, _ = env.reset(seed=seed + epoch)
        done = False
        episode_reward = 0.0
        episode_length = 0
        losses_in_episode = []

        while not done:
            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = _obs_to_tensor_single(state, DEVICE) 
                    q = agent(s)  
                    action = int(q.argmax(dim=1).item())

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            state_u8 = np.array(state, dtype=np.uint8)
            next_state_u8 = np.array(next_state, dtype=np.uint8)
            buffer.add((state_u8, action, float(reward), next_state_u8, int(done)))

            episode_reward += float(reward)
            episode_length += 1

            if buffer.real_size >= mini_batch_size:
                states, actions, rewards, next_states, dones = buffer.sample()
                dones_f = dones.float()
                with torch.no_grad():
                    if enable_double_dqn:
                        best_actions = agent(next_states).argmax(dim=1)  # (B,)
                        next_q = target_agent(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                    else:
                        next_q = target_agent(next_states).max(dim=1)[0]
                    target_q = rewards + (1.0 - dones_f) * discount_factor_g * next_q
                current_q = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = loss_fn(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                losses_in_episode.append(float(loss.detach().cpu()))
                sync_counter += 1

                if sync_counter >= network_sync_rate:
                    target_agent.load_state_dict(agent.state_dict())
                    sync_counter = 0

            state = next_state

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        avg_loss = float(np.mean(losses_in_episode)) if losses_in_episode else 0.0
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        avg_reward_100 = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0

        log_performance(
            epoch=epoch,
            avg_reward=episode_reward,
            loss=avg_loss,
            path=f"flappy_bird_cnn_dueling_dqn/{time_str}/train_performance_log.csv",
        )

        if epoch % 10 == 0:
            print(f"Episode {epoch}/{training_episodes} | eps={epsilon:.3f}")
            print(f"  Reward: {episode_reward:.2f} | Len: {episode_length}")
            print(f"  Avg Reward (last 100): {avg_reward_100:.2f} | Avg Loss: {avg_loss:.6f}")

        if avg_reward_100 > best_avg_reward and len(episode_rewards) >= 20:
            best_avg_reward = avg_reward_100
            save_checkpoint(
                model=agent,
                optimizer=optimizer,
                epoch=epoch,
                avg_reward=avg_reward_100,
                path="flappy_bird_cnn_dueling_dqn_checkpoints",
            )
            print(f"  New best model saved! Avg reward(100): {best_avg_reward:.2f}")

    save_checkpoint(
        model=agent,
        optimizer=optimizer,
        epoch=training_episodes,
        avg_reward=best_avg_reward if episode_rewards else 0.0,
        path="flappy_bird_cnn_dueling_dqn_checkpoints",
    )

    plot_rewards(
        log_dir="flappy_bird_cnn_dueling_dqn",
        window=10,
        save_path=f"flappy_bird_cnn_dueling_dqn/final_training_results_{time_str}.png",
    )

    env.close()
    print("Training finished...")
    print(f"Best average reward (last 100): {best_avg_reward:.2f}")
    print(f"Total episodes completed: {len(episode_rewards)}")


if __name__ == "__main__":
    train_cnn_dueling_dqn()