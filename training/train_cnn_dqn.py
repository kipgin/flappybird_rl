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
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.cnn import DQN_CNN
from algorithm.experience_replay import ReplayBufferCNN , PrioritizedReplayBufferCNN
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


def make_env(env_id: str, env_make_params: dict, num_frames: int, seed: int, rank: int, frame_skip: int = 1):
    def thunk():
        env_make_params_ = dict(env_make_params or {})
        env_make_params_.setdefault("use_lidar", False)

        env = gym.make(env_id, render_mode="rgb_array", disable_env_checker=True, **env_make_params_)
        env = FixFlappyStepAPI(env)

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

        env.action_space.seed(seed + rank)
        return env

    return thunk


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

def _obs_to_tensor_batch(obs_batch, device: torch.device) -> torch.Tensor:
    arr = np.asarray(obs_batch, dtype=np.uint8)
    x = torch.as_tensor(arr, device="cpu")
    x = x.float().div_(255.0)
    return x.to(device, non_blocking=True)

def train_cnn_dqn():
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
    max_grad_norm = float(hp.get("max_grad_norm", 10.0))
    enable_double_dqn = bool(hp.get("enable_double_dqn", False))
    enable_dueling_dqn = bool(hp.get("enable_dueling_dqn", False))

    hidden_dim = int(hp.get("hidden_dim", 128))
    # training_episodes = int(hp.get("training_episodes", 1_000_000))

    num_envs = int(hp.get("num_envs", 8))
    total_timesteps = int(hp.get("total_timesteps", 5_000_000))
    learning_starts = int(hp.get("learning_starts", 20_000))
    train_freq = int(hp.get("train_freq", 1))          
    gradient_steps = int(hp.get("gradient_steps", 1))  
    vector_mode = str(hp.get("vector_mode", "sync")).lower()  


    use_per = bool(hp.get("use_per", False))
    per_alpha = float(hp.get("per_alpha", 0.6))
    per_beta0 = float(hp.get("per_beta0", 0.4))
    per_eps = float(hp.get("per_eps", 1e-6))


    # num_frames = int(hp.get("frame_stack", 4))
    # env_make_params = hp.get("env_make_params", {})


    num_frames = int(hp.get("frame_stack", 4))
    env_make_params = hp.get("env_make_params", {})
    frame_skip = int(hp.get("frame_skip", 1)) 

    np.random.seed(seed)
    torch.manual_seed(seed)

    # env = make_env(env_id, env_make_params, num_frames=num_frames, seed=seed,frame_skip=frame_skip)
    # VecCls = AsyncVectorEnv if vector_mode == "async" else SyncVectorEnv
    envs = AsyncVectorEnv([make_env(env_id, env_make_params, num_frames=num_frames,frame_skip=frame_skip, seed=seed,rank =i ) for i in range(num_envs)])


    # action_dim = env.action_space.n
    # obs0, _ = env.reset(seed=seed)
    # obs0_arr = np.array(obs0)
    # obs_shape = tuple(obs0_arr.shape)


    obs, infos = envs.reset(seed=seed)
    action_dim = envs.single_action_space.n
    obs_shape = tuple(np.asarray(obs).shape[1:])

    ep_ret = np.zeros(num_envs, dtype=np.float32)
    ep_len = np.zeros(num_envs, dtype=np.int32)

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
    

    
    # buffer = ReplayBufferCNN(
    #     state_shape=obs_shape,
    #     action_size=action_dim,
    #     buffer_size=replay_memory_size,
    #     mini_batch_size=mini_batch_size,
    #     device=DEVICE,  
    # )

    #dung uniform replaybuffercnn
    if not use_per:
        buffer = ReplayBufferCNN(
            state_shape=obs_shape,
            action_size=action_dim,
            buffer_size=replay_memory_size,
            mini_batch_size=mini_batch_size,
            device=DEVICE,
        )

    #dung PER
    else:
        buffer = PrioritizedReplayBufferCNN(
            state_shape=obs_shape,
            action_size=action_dim,
            buffer_size=replay_memory_size,
            mini_batch_size=mini_batch_size,
            device=DEVICE,
            alpha=per_alpha,
            beta=per_beta0,
            eps=per_eps,
        )



    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate_a)
    loss_fn = torch.nn.MSELoss()

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_every_steps = 10000
    ckpt_every_steps =  200000
    best_avg_reward = -float("inf")
    episode_rewards = []
    episode_lengths = []


    last_loss_value = 0.0


    epsilon = epsilon_init
    sync_counter = 0

    print(f"Bat dau train CNN+DQN voi game: {env_id}")
    print(f"Episodes: {total_timesteps}, Replay: {replay_memory_size}, Batch: {mini_batch_size}, Device: {DEVICE}")

    global_step = 0

    exploration_steps = int(total_timesteps * 0.2)

    while global_step < total_timesteps:
        #epsilon-greedy
        rand_mask = (np.random.rand(num_envs) < epsilon)
        actions = np.random.randint(action_dim, size=num_envs, dtype=np.int64)
        if not np.all(rand_mask):
            with torch.no_grad():
                obs_dev = _obs_to_tensor_batch(obs, DEVICE)
                q = agent(obs_dev)
                greedy = q.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            actions[~rand_mask] = greedy[~rand_mask]

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        dones = np.logical_or(terminated, truncated)

        #next_obs tra ve la state cua new episode. Can lay final_observation tu infos
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            # infos["_final_observation"] la mask boolean cho biet env nao da reset
            for idx, is_final in enumerate(infos.get("_final_observation", [])):
                if is_final:
                    real_next_obs[idx] = infos["final_observation"][idx]

        buffer.add_batch(
            states_u8=np.asarray(obs, dtype=np.uint8),
            actions=actions,
            rewards=rewards,
            next_states_u8=np.asarray(real_next_obs, dtype=np.uint8),
            dones=terminated.astype(np.uint8),
        )

        ep_ret += rewards.astype(np.float32)
        ep_len += 1
        done_idxs = np.where(dones)[0]
        if done_idxs.size > 0:
            for i in done_idxs.tolist():
                episode_rewards.append(float(ep_ret[i]))
                episode_lengths.append(int(ep_len[i]))
                ep_ret[i] = 0.0
                ep_len[i] = 0

            # # thuc ra VectorEnv co san roi nay khong can
            # reset_out = envs.env_method("reset", indices=done_idxs.tolist(), seed=[seed + 10_000 + global_step + int(i) for i in done_idxs.tolist()])
            # # reset_out is list of tuples [(obs, info), ...]
            # for j, i in enumerate(done_idxs.tolist()):
            #     next_obs[i] = reset_out[j][0]

        obs = next_obs
        global_step += num_envs

      
        if global_step < exploration_steps:
            epsilon = epsilon_init - (epsilon_init - epsilon_min) * (global_step / exploration_steps)
        else:
            epsilon = epsilon_min

        #updating learner
        if buffer.real_size >= mini_batch_size and global_step >= learning_starts and (global_step // num_envs) % train_freq == 0:
            agent.train()
            for _ in range(max(1, gradient_steps)):
                if not use_per:
                    states, actions_t, rewards_t, next_states, dones_t = buffer.sample()
                    loss = agent.td_loss(
                        target_net=target_agent,
                        states=states,
                        actions=actions_t,
                        rewards=rewards_t,
                        next_states=next_states,
                        dones=dones_t,
                        gamma=discount_factor_g,
                        double_dqn=enable_double_dqn,
                        loss_fn=loss_fn,
                    )
                else:
                    beta = min(1.0, per_beta0 + (1.0 - per_beta0) * (global_step / float(total_timesteps)))

                    states, actions_t, rewards_t, next_states, dones_t, idxs, weights = buffer.sample(beta=beta)
                    td_err = agent.td_errors(
                        target_net=target_agent,
                        states=states,
                        actions=actions_t,
                        rewards=rewards_t,
                        next_states=next_states,
                        dones=dones_t,
                        gamma=discount_factor_g,
                        double_dqn=enable_double_dqn,
                    )
                    loss = (weights * td_err.pow(2)).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
                

                last_loss_value = float(loss.detach().cpu())

                if use_per:
                    new_p = (td_err.detach().abs().cpu().numpy() + per_eps)
                    buffer.update_priorities(idxs, new_p)

                sync_counter += 1
                if sync_counter >= network_sync_rate:
                    target_agent.load_state_dict(agent.state_dict())
                    sync_counter = 0

        # logging
        if episode_rewards:
            avg_reward = float(np.mean(episode_rewards[-100:]))
            avg_length = float(np.mean(episode_lengths[-100:]))
            best_avg_reward = max(best_avg_reward, avg_reward)
        else:
            avg_reward = 0.0
            avg_length = 0.0

        # if global_step % log_every_steps == 0:
        log_performance(
            epoch=global_step, 
            avg_reward=avg_reward,
            loss=last_loss_value,
            path=f"flappy_bird_cnn_dqn/{time_str}/train_performance_log.csv",
        )
        if global_step % log_every_steps == 0:
            print(f"Steps={global_step} | eps={epsilon:.3f} | Episodes={len(episode_rewards)}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Length (last 100): {avg_length:.2f}")
            print(f"  Loss: {last_loss_value:.6f}")
            print(f"  Best Avg Reward: {best_avg_reward:.2f}")

        if global_step % ckpt_every_steps == 0 and global_step > 0:
            save_checkpoint(
                model=agent,
                optimizer=optimizer,
                epoch=global_step,
                avg_reward=avg_reward,
                path="flappy_bird_cnn_dqn_checkpoints",
            )

    save_checkpoint(
        model=agent,
        optimizer=optimizer,
        epoch=global_step,
        avg_reward=best_avg_reward if episode_rewards else 0.0,
        path="flappy_bird_cnn_dqn_checkpoints",
    )

    plot_rewards(
        log_dir="flappy_bird_cnn_dqn",
        window=10,
        save_path=f"flappy_bird_cnn_dqn/final_training_results_{time_str}.png",
    )

    envs.close()
    print("Training finished...")
    print(f"Best average reward (last 100): {best_avg_reward:.2f}")
    print(f"Total episodes completed: {len(episode_rewards)}")

if __name__ == "__main__":
    train_cnn_dqn()