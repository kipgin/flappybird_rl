import torch
import torch.nn as nn

from algorithm.dqn import DQN
from algorithm.policy_gradient import PolicyGradient
from algorithm.ppo import PPO

#doc file hyperparmeters.yml
def _make_cnn_cfg(cfg: dict) -> dict:
    return {
        "input_channels": int(cfg.get("input_channels", 4)),
        "conv_channels": cfg.get("conv_channels", [32, 64, 64]),
        "kernel_sizes": cfg.get("kernel_sizes", [8, 4, 3]),
        "strides": cfg.get("strides", [4, 2, 1]),
        "paddings": cfg.get("paddings", [0, 0, 0]),
        "hidden_size": int(cfg.get("hidden_size", 512)),
        "activation": str(cfg.get("activation", "relu")).lower(),
        "head_hidden": int(cfg.get("head_hidden", 256)),
    }


class CNNEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        in_channels = int(cfg["input_channels"])
        conv_channels = cfg["conv_channels"]
        kernel_sizes = cfg["kernel_sizes"]
        strides = cfg["strides"]
        paddings = cfg["paddings"]
        hidden_size = int(cfg["hidden_size"])
        activation = str(cfg.get("activation", "relu")).lower()

        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "silu":
            act_fn = nn.SiLU

        layers = []
        for out_ch, k, s, p in zip(conv_channels, kernel_sizes, strides, paddings):
            layers.append(nn.Conv2d(in_channels, out_ch, kernel_size=k, stride=s, padding=p))
            layers.append(act_fn())
            in_channels = out_ch

        self.hidden_size = hidden_size
        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(hidden_size)
        self.act = act_fn()

    # def _init_fc(self, x: torch.Tensor):
    #     with torch.no_grad():
    #         n_flatten = self.conv(x).view(x.size(0), -1).size(1)
    #     self.fc = nn.Linear(n_flatten, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        else:
            x = x.float()
        # if self.fc is None:
        #     self._init_fc(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.act(x)
        return x

class DQN_CNN(DQN):
    def __init__(self, obs_shape, action_dim, hidden_dim, enable_dueling_dqn=True, cfg=None):
        ccfg = _make_cnn_cfg(cfg)
        super().__init__(state_dim=int(ccfg["hidden_size"]), action_dim=action_dim, hidden_dim=hidden_dim, enable_dueling_dqn=enable_dueling_dqn)
        self.encoder = CNNEncoder(ccfg)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)
        return super().forward(features)


class PPO_CNN(PPO):
    def __init__(self, clip_coef, vf_coef, ent_coef, lr, max_grad_norm, update_epochs, num_minibatches,
                 obs_shape, action_dim, layer_size, cfg=None):
        ccfg = _make_cnn_cfg(cfg)

        feature_dim = int(ccfg["hidden_size"])
        super().__init__(clip_coef, vf_coef, ent_coef, lr, max_grad_norm, update_epochs, num_minibatches,
                         state_dim=feature_dim, action_dim=action_dim, layer_size=layer_size)
        self.encoder_actor = CNNEncoder(ccfg)
        self.encoder_critic = CNNEncoder(ccfg)

    def get_value(self, obs: torch.Tensor):
        features = self.encoder_critic(obs)
        return self.critic(features).flatten()

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        features_a = self.encoder_actor(obs)
        logits = self.actor(features_a)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        features_v = self.encoder_critic(obs)
        value = self.critic(features_v).flatten()
        return action, log_prob, entropy, value


class PolicyGradient_CNN(PolicyGradient):
    def __init__(self, vf_coef, ent_coef, lr, max_grad_norm, obs_shape, action_dim, hidden_dim, use_baseline=True, cfg=None):
        ccfg = _make_cnn_cfg(cfg)
        feature_dim = int(ccfg["hidden_size"])
        super().__init__(vf_coef, ent_coef, lr, max_grad_norm, state_dim=feature_dim, action_dim=action_dim, hidden_dim=hidden_dim, use_baseline=use_baseline)
        self.encoder_actor = CNNEncoder(ccfg)
        self.encoder_critic = CNNEncoder(ccfg)

    def get_action_and_value(self, obs: torch.Tensor, actions=None):
        features = self.encoder_actor(obs)
        logits = self.actor(features)
        probs = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = probs.sample()
        log_probs = probs.log_prob(actions)
        entropy = probs.entropy()

        v = self.critic(self.encoder_critic(obs)).squeeze(-1)
        return actions, log_probs, entropy, v