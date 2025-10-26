# import torch

# class RolloutBuffer:
#     def __init__(self, num_steps, num_envs, state_shape, action_shape, device):
#         self.num_steps = num_steps
#         self.num_envs = num_envs
#         self.device = device
        
#         self.obs = torch.zeros((num_steps, num_envs,state_shape),dtype=torch.float32, device=device)
#         self.actions = torch.zeros((num_steps, num_envs,action_shape), dtype = torch.int64, device=device)
#         self.logprobs = torch.zeros((num_steps, num_envs),dtype = torch.float32,device=device)
#         self.rewards = torch.zeros((num_steps, num_envs),dtype = torch.float32, device=device)
#         self.dones = torch.zeros((num_steps, num_envs),dtype = torch.int64, device=device)
#         self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32,device=device)
        
#         self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
#         self.returns = torch.zeros((num_steps, num_envs),  dtype=torch.float32,device=device)
        
#         self.step = 0

#     def add(self, obs, action, logprob, reward, done, value):
#         self.obs[self.step].copy_(obs)
#         self.actions[self.step].copy_(action)
#         self.logprobs[self.step].copy_(logprob)
#         self.rewards[self.step].copy_(reward)
#         self.dones[self.step].copy_(done)
#         self.values[self.step].copy_(value)
#         self.step = (self.step + 1) % self.num_steps

#     # def compute_returns_pg(self,last_value,last_donw,)
#     def compute_returns_and_advantages(self, last_value, last_done, gamma, gae_lambda):
#         last_gae_lam = 0
#         for t in reversed(range(self.num_steps)):
#             if t == self.num_steps - 1:
#                 next_non_terminal = 1.0 - last_done
#                 next_value = last_value
#             else:
#                 next_non_terminal = 1.0 - self.dones[t + 1]
#                 next_value = self.values[t + 1]
#             delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
#             last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
#             self.advantages[t] = last_gae_lam
#         self.returns = self.advantages + self.values

#     def get(self):
#         batch = {
#             'obs': self.obs.reshape(-1, *self.obs.shape[2:]),
#             'actions': self.actions.reshape(-1, *self.actions.shape[2:]),
#             'logprobs': self.logprobs.flatten(),
#             'returns': self.returns.flatten(),
#             'advantages': self.advantages.flatten(),
#             'values': self.values.flatten(),
#         }
#         return batch
#     def len(self):
#         return len(self.obs)

#     def clear(self):
#         self.step = 0
#         self.advantages.zero_()
#         self.returns.zero_()
#         self.obs = torch.zeros((self.num_steps, self.num_envs) + self.state_shape, dtype=torch.float32, device=self.device)
#         self.actions = torch.zeros((self.num_steps, self.num_envs) + self.action_shape, dtype = torch.int64, device=self.device)
#         self.logprobs = torch.zeros((self.num_steps, self.num_envs), dtype = torch.float32, device=self.device)
#         self.rewards = torch.zeros((self.num_steps, self.num_envs), dtype = torch.float32, device=self.device)
#         self.dones = torch.zeros((self.num_steps, self.num_envs), dtype = torch.int64, device=self.device)
#         self.values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
import torch

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, state_shape, action_shape, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        
        self.obs = torch.zeros((num_steps, num_envs, state_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.int64, device=device)  
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.int64, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        
        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        
        self.step = 0
        self.state_shape = state_shape  

    def add(self, obs, action, logprob, reward, done, value):
        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(action)
        self.logprobs[self.step].copy_(logprob)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(done)
        self.values[self.step].copy_(value)
        self.step = (self.step + 1) % self.num_steps

    def compute_returns_and_advantages(self, last_value, last_done, gamma, gae_lambda):
        last_gae_lam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        self.returns = self.advantages + self.values

    def get(self):
        batch = {
            'obs': self.obs.reshape(-1, self.state_shape), 
            'actions': self.actions.flatten(), 
            'logprobs': self.logprobs.flatten(),
            'returns': self.returns.flatten(),
            'advantages': self.advantages.flatten(),
            'values': self.values.flatten(),
        }
        return batch
        
    def len(self):
        return len(self.obs)

    def clear(self):
        self.step = 0
        self.advantages.zero_()
        self.returns.zero_()
        self.obs = torch.zeros((self.num_steps, self.num_envs, self.state_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs), dtype=torch.int64, device=self.device)  # Changed
        self.logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.int64, device=self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)