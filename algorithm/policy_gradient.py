import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.output(x)


class PolicyGradient(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, use_baseline=False):
        super(PolicyGradient, self).__init__()        
        self.use_baseline = use_baseline
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        if self.use_baseline:
            self.critic = ValueNetwork(state_dim, hidden_dim)

    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        state_value = None
        if self.use_baseline:
            state_value = self.critic(state).squeeze() 

        return action.item(), log_prob, state_value

    def evaluate(self, states, actions):
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy() 
        state_values = None
        if self.use_baseline:
            state_values = self.critic(states).squeeze()
            
        return log_probs, state_values, entropy