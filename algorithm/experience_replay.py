import torch
import numpy as np

device = 'cpu'

class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size,mini_batch_size):
        # state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float32)
        self.action = torch.empty(buffer_size,  dtype=torch.int64)
        self.reward = torch.empty(buffer_size, dtype=torch.float32)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float32)
        self.done = torch.empty(buffer_size, dtype=torch.int64)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.mini_batch_size = mini_batch_size
        # self.device = device


    def add(self, transition):
        state, action, reward, next_state, done = transition
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self):
        # mini_batch_size = self.mini_batch_size
        assert self.real_size >= self.mini_batch_size
        sample_idxs = np.random.choice(self.real_size, self.mini_batch_size, replace=False)
        mini_batch = (
            self.state[sample_idxs].to(device),
            self.action[sample_idxs].to(device),
            self.reward[sample_idxs].to(device),
            self.next_state[sample_idxs].to(device),
            self.done[sample_idxs].to(device)
        )
        return mini_batch