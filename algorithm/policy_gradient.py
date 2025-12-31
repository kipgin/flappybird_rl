import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import *

# class ValueNetwork(nn.Module):
#     def __init__(self, state_dim, hidden_dim):
#         super(ValueNetwork, self).__init__()
#         self.critic = nn.Sequential(
#             layer_init(nn.Linear(state_dim, hidden_dim)),
#             nn.Tanh(), 
#             layer_init(nn.Linear(hidden_dim, hidden_dim)),
#             nn.Tanh(),  
#             layer_init(nn.Linear(hidden_dim, 1), std=1.0) 
#         )


class PolicyGradient(nn.Module):
    def __init__(self, vf_coef, ent_coef, lr, max_grad_norm, state_dim, action_dim, hidden_dim, use_baseline=True):
        super().__init__()
        self.use_baseline = bool(use_baseline)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

        self.critic = None
        if self.use_baseline:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(state_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, 1), std=1.0),
            )

        # FIX: optimizer must always exist
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        logits = self.actor(state)
        return logits.argmax(dim=0)

    def get_action_and_value(self, states, actions=None):
        logits = self.actor(states)
        probs = Categorical(logits=logits)
        if actions is None:
            actions = probs.sample()
        log_probs = probs.log_prob(actions)
        entropy = probs.entropy()

        if self.use_baseline:
            values = self.critic(states).squeeze(-1)
        else:
            values = torch.zeros(states.shape[0], device=states.device, dtype=torch.float32)

        return actions, log_probs, entropy, values

    def update(self, buffer):
        model_device = next(self.parameters()).device  # FIX: no global device

        batch = buffer.get()
        b_obs = batch["obs"].to(model_device)
        b_actions = batch["actions"].to(model_device)
        b_advantages = batch["advantages"].to(model_device)
        b_returns = batch["returns"].to(model_device)

        _, new_log_probs, entropy, new_values = self.get_action_and_value(b_obs, b_actions)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        pg_loss = -(b_advantages * new_log_probs).mean()

        if self.use_baseline:
            v_loss = ((b_returns - new_values) ** 2).mean()
        else:
            v_loss = torch.tensor(0.0, device=model_device)

        entropy_loss = entropy.mean()
        loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

        self.optimizer.zero_grad()
        loss.backward()
        foreach = False if next(self.parameters()).device.type == "xpu" else True

        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm,foreach=foreach)
        self.optimizer.step()

        return float(loss.detach().cpu())
