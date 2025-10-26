import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.dqn import DQN
from algorithm.policy_gradient import PolicyGradient
from algorithm.ppo import PPO

class CNNEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg["input_channels"]
        conv_channels = cfg["conv_channels"]
        kernel_sizes = cfg["kernel_sizes"]
        strides = cfg["strides"]
        paddings = cfg["paddings"]
        hidden_size = cfg["hidden_size"]
        activation = cfg.get("activation", "relu").lower()

        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "silu":
            act_fn = nn.SiLU

        #convolutional stack
        conv_layers = []
        for out_channels, k, s, p in zip(conv_channels, kernel_sizes, strides, paddings):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, k, stride=s, padding=p))
            conv_layers.append(act_fn())
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        #lazy init
        self.flatten = nn.Flatten()
        self.fc = None
        self.hidden_size = hidden_size
        self.act = act_fn()

    def _init_fc(self, x):
        with torch.no_grad():
            n_flatten = self.conv(x).view(x.size(0), -1).size(1)
        self.fc = nn.Linear(n_flatten, self.hidden_size)

    def forward(self, x):
        x = x / 255.0 if x.dtype == torch.uint8 else x
        if self.fc is None:
            self._init_fc(x)

        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.act(x)
        return x

class DQN_CNN(DQN):
    def __init__(self, state_dim, action_dim, hidden_dim, enable_dueling_dqn,cfg):
        super().__init__(state_dim, action_dim, hidden_dim, enable_dueling_dqn)
        self.cnn = CNNEncoder(cfg)
    def out(self,x):
        x=self.forward(x)
        x = self.cnn.forward(x)
        return x

class 
        