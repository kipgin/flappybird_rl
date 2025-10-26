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
device = 'cpu'

class PolicyGradient(nn.Module):
    def __init__(self,vf_coef,ent_coef,lr,max_grad_norm, state_dim, action_dim, hidden_dim,use_baseline=True):
        super(PolicyGradient, self).__init__()        
        self.use_baseline = use_baseline
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim,hidden_dim)),
            nn.Tanh(),  
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        )
        if self.use_baseline:
            self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(), 
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),  
            layer_init(nn.Linear(hidden_dim, 1), std=1.0) 
        )
            self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)

    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        state_value = None
        if self.use_baseline:
            state_value = self.critic(state).squeeze(-1) 

        return action.item(), log_prob, state_value

    def get_action_and_value(self,states,actions = None):
        logits = self.actor(states)
        probs = torch.distributions.Categorical(logits=logits)
        if actions is None :
            actions = probs.sample()
        log_probs = probs.log_prob(actions)
        entropy = probs.entropy()
        values = self.critic(states).squeeze(-1)
        return actions, log_probs, entropy, values

    def evaluate(self, states, actions):
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy() 
        state_values = None
        if self.use_baseline:
            state_values = self.critic(states).squeeze(-1)
            
        return log_probs, state_values, entropy
    
    def update(self,buffer):
        batch = buffer.get()
        b_obs = batch['obs'].to(device)
        b_logprobs = batch['logprobs'].to(device)
        b_actions = batch['actions'].to(device)
        b_advantages = batch['advantages'].to(device)
        b_returns = batch['returns'].to(device)
        b_values = batch['values'].to(device)

        _,new_log_probs,entropy,new_values=self.get_action_and_value(b_obs,b_actions)
        b_advantages = (b_advantages - b_advantages.mean())/(b_advantages.std() + 1e-8)

        pg_loss = -(b_advantages * new_log_probs).mean()
        v_loss = ((b_returns - new_values)**2).mean()

        entropy_loss = entropy.mean()

        loss = pg_loss - self.ent_coef*entropy_loss + self.vf_coef * v_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        
        # buffer.clear()
