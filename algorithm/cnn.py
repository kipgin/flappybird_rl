import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from algorithm.dqn import DQN
from algorithm.policy_gradient import PolicyGradient
from algorithm.ppo import PPO
from utils import layer_init

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

        self.conv = nn.Sequential(*[
            layer_init(layer) if isinstance(layer, nn.Conv2d) else layer 
            for layer in layers
        ])
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, int(cfg["input_channels"]), 84, 84)
            n_flatten = self.conv(dummy_input).view(1, -1).shape[1]
            
        self.fc = layer_init(nn.Linear(n_flatten, hidden_size))
        self.act = act_fn()

    def _init_fc(self, x: torch.Tensor):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        else:
            x = x.float()
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
        
        feature_dim = int(ccfg["hidden_size"])
        if self.enable_dueling_dqn:
            self.v_head = layer_init(nn.Linear(feature_dim, 1))
            self.a_head = layer_init(nn.Linear(feature_dim, action_dim))
        else:
            self.q_head = layer_init(nn.Linear(feature_dim, action_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.encoder(obs)
        
        if self.enable_dueling_dqn:
            V = self.v_head(x)
            A = self.a_head(x)
            Q = V + A - A.mean(dim=1, keepdim=True)
        else:
            Q = self.q_head(x)
            
        return Q


class PPO_CNN(PPO):
    def __init__(self, clip_coef, vf_coef, ent_coef, lr, max_grad_norm, update_epochs, num_minibatches,
                 obs_shape, action_dim, layer_size, cfg=None):
        ccfg = _make_cnn_cfg(cfg)

        # update_epochs = int((cfg or {}).get("update_epochs", 1))
        # num_minibatches = int((cfg or {}).get("num_minibatches", 1))

        feature_dim = int(ccfg["hidden_size"])
        super().__init__(clip_coef, 
                         vf_coef, 
                         ent_coef, 
                         lr, 
                         max_grad_norm, 
                         update_epochs, 
                         num_minibatches,
                         state_dim=feature_dim, action_dim=action_dim, layer_size=layer_size)
        
        self.network = CNNEncoder(ccfg)
        
        self.actor = layer_init(nn.Linear(feature_dim, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(feature_dim, 1), std=1.0)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)
    
    def get_value(self, obs: torch.Tensor):
        hidden = self.network(obs)
        return self.critic(hidden).flatten()

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        hidden = self.network(obs)
        logits = self.actor(hidden)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        value = self.critic(hidden).flatten()
        
        return action, log_prob, entropy, value


class PolicyGradient_CNN(PolicyGradient):
    def __init__(self, vf_coef, ent_coef, lr, max_grad_norm, obs_shape, action_dim, hidden_dim, use_baseline=True, cfg=None):
        ccfg = _make_cnn_cfg(cfg)
        feature_dim = int(ccfg["hidden_size"])
        update_epochs = int((cfg or {}).get("update_epochs", 1))
        num_minibatches = int((cfg or {}).get("num_minibatches", 1))

        super().__init__(vf_coef, 
                         ent_coef, 
                         lr, 
                         max_grad_norm, 
                         state_dim=feature_dim, 
                         action_dim=action_dim, 
                         hidden_dim=hidden_dim, 
                         use_baseline=use_baseline,
                         update_epochs=update_epochs,
                         num_minibatches=num_minibatches
                         )
        
        self.network = CNNEncoder(ccfg)

        
        self.actor = layer_init(nn.Linear(feature_dim, action_dim), std=0.01)
        if self.use_baseline:
            self.critic = layer_init(nn.Linear(feature_dim, 1), std=1.0)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def get_action_and_value(self, obs: torch.Tensor, actions=None):
        features = self.network(obs)
        return super().get_action_and_value(features, actions)