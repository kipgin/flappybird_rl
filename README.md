# Project: Introduction to AI - IT3160 (HUST)

**Members:**
- Nguyễn Ngọc Trung - 20230074 - ITTN K68

- Nguyễn Hà Anh Tuấn - 20230076 - ITTN K68

- Nguyễn Công Sơn - 20230061 - ITTN K68

- Vũ Đức Tâm - 20230064 - ITTN K68

---

# Flappy Bird Reinforcement Learning

Train and deploy RL agents to play Flappy Bird using PyTorch on Windows.

## Requirements

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `torch>=1.12.0` ([Install with CUDA support](https://pytorch.org/get-started/locally/))
- `gymnasium>=0.28.0`
- `flappy_bird_gymnasium`
- `numpy`, `pandas`, `matplotlib`, `PyYAML`

## Quick Start

### 1. Play the Game
```bash
flappy_bird_gymnasium                    # Human play with LiDAR
flappy_bird_gymnasium --mode random      # Watch random agent
```

### 2. Training

Navigate to training directory:
```bash
cd training
```

Train your agent:
```bash
python train_dqn.py                # Train DQN variants
python train_policy_gradient.py   # Train Policy Gradient
python train_ppo.py                # Train PPO
```

**Training outputs:**
- Logs & plots: `training/flappy_bird_{algorithm}/`
- Model weights: `training/flappy_bird_{algorithm}_checkpoints/best_model.pth`

### 3. Inference (Watch Trained Agents)

Navigate to inference directory:
```bash
cd inference
```

| Algorithm | Command | Config |
|-----------|---------|--------|
| **DQN** | `python infer_dqn.py` | `enable_dueling_dqn: False`<br>`enable_double_dqn: False` |
| **Dueling DQN** | `python infer_dqn.py` | `enable_dueling_dqn: True` |
| **Double DQN** | `python infer_dqn.py` | `enable_double_dqn: True` |
| **Policy Gradient** | `python infer_policy_gradient.py` | Default |
| **PPO** | `python infer_ppo.py` | Default |

## Configuration

Edit `hyperparameters.yml` to customize:

- **Learning rates:** `learning_rate_a`, `lr`
- **Network architecture:** `hidden_dim`, `fc1_nodes`
- **DQN variants:** `enable_dueling_dqn`, `enable_double_dqn`
- **Exploration:** `epsilon_init`, `epsilon_decay`, `epsilon_min`
- **Parallel training:** `num_envs` (Policy Gradient & PPO)
- **GAE parameters:** `gamma`, `gae_lambda`

## Implementation Details

- **Framework:** PyTorch
- **Observation space:** Numerical features (bird position, velocity, pipe distances)
- **Action space:** Discrete (0=idle, 1=flap)
- **Policy Gradient & PPO:** 
    - 4 parallel environments for efficient data collection
    - Generalized Advantage Estimation (GAE)

## Coming Soon (done)

🚀 **Vision-based agents:** CNN architecture for raw pixel observations

## 4. Inference (CNN Agents + Video Recording)

> These scripts run **vision-based (CNN)** agents that use **rendered frames** as observations (resize + grayscale + frame stack).
> Make sure you have a trained `.pth` checkpoint.

Navigate to inference directory:
```bash
cd inference
```

### PPO-CNN
Deterministic (recommended for evaluation):
```bash
python infer_cnn.py --algo cnn_ppo --model-path ..\training\flappy_bird_cnn_ppo_checkpoints\{model_file_name}.pth --episodes 5 --deterministic
```

Record video:
```bash
python infer_cnn.py --algo cnn_ppo --model-path ..\training\flappy_bird_cnn_ppo_checkpoints\{model_file_name}.pth --episodes 3 --deterministic --record-video-dir ..\videos\ppo_cnn
```

### Policy Gradient - CNN
Deterministic:
```bash
python infer_cnn.py --algo cnn_policy_gradient --model-path ..\training\flappy_bird_cnn_policy_gradient_checkpoints\{model_file_name}.pth --episodes 5 --deterministic
```

Record video:
```bash
python infer_cnn.py --algo cnn_policy_gradient --model-path ..\training\flappy_bird_cnn_policy_gradient_checkpoints\{model_file_name}.pth --episodes 3 --deterministic --record-video-dir ..\videos\pg_cnn
```

### DQN-CNN
DQN is naturally greedy at inference (argmax Q), so `--deterministic` is recommended:
```bash
python infer_cnn.py --algo cnn_dqn --model-path ..\training\flappy_bird_cnn_dqn_checkpoints\{model_file_name}.pth --episodes 5 --deterministic
```

Record video:
```bash
python infer_cnn.py --algo cnn_dqn --model-path ..\training\flappy_bird_cnn_dqn_checkpoints\{model_file_name}.pth --episodes 3 --deterministic --record-video-dir ..\videos\dqn_cnn
```

### Notes
- Use `--config-key` to pick a specific section in `hyperparameters.yml` (defaults are:
  `ppo_flappybird`, `policy_gradient_flappybird`, `flappybird1`).
- If you run into Intel XPU issues, you can force CPU inference:
```bash
python infer_cnn.py --algo cnn_ppo --model-path ..\flappy_bird_cnn_ppo_checkpoints\best_model.pth --force-cpu --episodes 3 --deterministic
```



## References

### Papers
1. [Playing Atari with Deep RL](https://arxiv.org/pdf/1312.5602) (DQN)
2. [Dueling Network Architectures](https://arxiv.org/pdf/1511.06581)
3. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461)
4. [High-Dimensional Continuous Control Using GAE](https://arxiv.org/pdf/1506.02438)
5. [Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347)
6. [Policy Gradient Methods](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

### Code & Tutorials
- [Flappy Bird Gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)
- [DQN PyTorch Tutorial](https://github.com/johnnycode8/dqn_pytorch)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course)



