import argparse
import os
import sys
import yaml
import numpy as np
import torch
import gymnasium as gym
import cv2
import time

from gymnasium import ObservationWrapper
from gymnasium.spaces import Box


from gymnasium.wrappers import GrayscaleObservation as _GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation as _FrameStack


import flappy_bird_gymnasium  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.cnn import PPO_CNN, PolicyGradient_CNN, DQN_CNN


def select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print(f"Using {torch.xpu.get_device_name(0)}")
        return torch.device("xpu")
    print("No cuda or xpu, using cpu")
    return torch.device("cpu")


def load_cfg(yaml_path: str, config_key: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        all_cfg = yaml.safe_load(f)
    return all_cfg[config_key]


class FixFlappyStepAPI(gym.Wrapper):
    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 6:
            obs, reward, terminated, truncated, info, _extra = out
            return obs, reward, terminated, truncated, info
        return out 
    
class ActionRepeat(gym.Wrapper):
    def __init__(self, env, skip: int = 4, fps: int = 30):
        super().__init__(env)
        self.skip = int(skip)
        self.fps = fps
        self.frame_duration = 1.0 / fps

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for _ in range(max(1, self.skip)):
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            
            frame = self.env.render()
            if frame is not None:
                view_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Flappy Bird", view_frame)
                if (cv2.waitKey(int(self.frame_duration * 1000)) & 0xFF) == ord("q"):
                    terminated = True

            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class RenderObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(512, 288, 3), dtype=np.uint8)

    def observation(self, obs):
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None. Ensure render_mode='rgb_array'.")
        return frame


def make_env(env_id: str, cfg: dict, seed: int, num_frames: int, record_video_dir: str | None):
    env_make_params = dict(cfg.get("env_make_params", {}) or {})
    env_make_params.setdefault("use_lidar", False)
    # env = gym.make(env_id, render_mode="rgb_array", **env_make_params)
    env = gym.make(env_id, render_mode="rgb_array", disable_env_checker=True, **env_make_params)
    
    env = FixFlappyStepAPI(env)

    frame_skip = int(cfg.get("frame_skip", 1))
    fps = env.metadata.get("render_fps", 30)
    env = ActionRepeat(env, skip=frame_skip, fps=fps)

    if record_video_dir:
        os.makedirs(record_video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, video_folder=record_video_dir, episode_trigger=lambda ep: True)
    env = RenderObservation(env)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    try:
        env = _GrayscaleObservation(env, keep_dim=False)
    except TypeError:
        env = _GrayscaleObservation(env)

    env = _FrameStack(env, num_frames)
    env.action_space.seed(seed)
    return env


def obs_to_tensor_single(obs, device: torch.device) -> torch.Tensor:
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


def load_checkpoint_into_model(model: torch.nn.Module, model_path: str, device: torch.device):
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def select_action(model, algo: str, obs_t: torch.Tensor, deterministic: bool) -> int:
    if algo == "cnn_dqn":
        q = model(obs_t)  
        return int(q.argmax(dim=1).item())
    if algo in ("cnn_ppo", "cnn_policy_gradient"):
        if deterministic:
            logits = model.actor(model.network(obs_t))
            return int(logits.argmax(dim=-1).item())
        action, _, _, _ = model.get_action_and_value(obs_t)
        return int(action.item())
    raise ValueError(f"Unknown algo: {algo}")


def build_model(algo: str, cfg: dict, obs_shape: tuple, action_dim: int):
    if algo == "cnn_dqn":
        hidden_dim = int(cfg.get("hidden_dim", 128))
        enable_dueling_dqn = bool(cfg.get("enable_dueling_dqn", False))
        return DQN_CNN(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            enable_dueling_dqn=enable_dueling_dqn,
            cfg=cfg,
        )
    if algo == "cnn_ppo":
        return PPO_CNN(
            clip_coef=float(cfg["clip_coef"]),
            vf_coef=float(cfg["vf_coef"]),
            ent_coef=float(cfg["ent_coef"]),
            lr=float(cfg["lr"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
            update_epochs=int(cfg["update_epochs"]),
            num_minibatches=int(cfg["num_minibatches"]),
            obs_shape=obs_shape,
            action_dim=action_dim,
            layer_size=int(cfg["layer_size"]),
            cfg=cfg,
        )

    if algo == "cnn_policy_gradient":
        return PolicyGradient_CNN(
            vf_coef=float(cfg["vf_coef"]),
            ent_coef=float(cfg["ent_coef"]),
            lr=float(cfg["lr"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_dim=int(cfg["hidden_dim"]),
            use_baseline=bool(cfg.get("use_baseline", True)),
            cfg=cfg,
        )

    raise ValueError(f"Unknown algo: {algo}")


def default_config_key_for_algo(algo: str) -> str:
    if algo == "cnn_ppo":
        return "ppo_flappybird"
    if algo == "cnn_policy_gradient":
        return "policy_gradient_flappybird"
    if algo == "cnn_dqn":
        return "flappybird1"
    raise ValueError(algo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["cnn_ppo", "cnn_policy_gradient", "cnn_dqn"], required=True)
    ap.add_argument("--model-path", type=str, required=True, help="Path to .pth checkpoint")
    ap.add_argument("--yaml", type=str, default=os.path.join("..", "hyperparameters.yml"))
    ap.add_argument("--config-key", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--record-video-dir", type=str, default=None)
    args = ap.parse_args()

    device = select_device(force_cpu=args.force_cpu)

    config_key = args.config_key or default_config_key_for_algo(args.algo)
    cfg = load_cfg(args.yaml, config_key)

    env_id = cfg.get("env_id", "FlappyBird-v0")
    num_frames = int(cfg.get("frame_stack", 4))

    env = make_env(env_id, cfg, seed=args.seed, num_frames=num_frames, record_video_dir=args.record_video_dir)

    obs, _ = env.reset(seed=args.seed)
    obs_shape = tuple(np.array(obs).shape)  
    action_dim = env.action_space.n

    model = build_model(args.algo, cfg, obs_shape=obs_shape, action_dim=action_dim)
    model = load_checkpoint_into_model(model, args.model_path, device)

    with torch.no_grad():
        dummy = torch.zeros((1, *obs_shape), device=device, dtype=torch.float32)
        if args.algo == "cnn_dqn":
            _ = model(dummy)
        else:
            _ = model.get_action_and_value(dummy)

    print(f"Infer algo={args.algo}, env={env_id}, obs_shape={obs_shape}, action_dim={action_dim}, device={device}")
    print(f"Model: {args.model_path}")
    print(f"Deterministic: {args.deterministic}")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_r = 0.0
        steps = 0

        for _ in range(args.max_steps):
            obs_t = obs_to_tensor_single(obs, device)
            action = select_action(model, args.algo, obs_t, deterministic=args.deterministic)
          
            obs, reward, terminated, truncated, info = env.step(action)

            total_r += float(reward)
            steps += 1

            if terminated or truncated:
                break

        print(f"Episode {ep}: reward={total_r:.2f}, length={steps}")

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()