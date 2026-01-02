import torch
import numpy as np
# from seg_tree import _MinSegmentTree , _SumSegmentTree
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.seg_tree import _SumSegmentTree,_MinSegmentTree

device = ''
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using {torch.cuda.get_device_name(0)}")

elif torch.xpu.is_available():
    device = torch.device("xpu")
    print(f"Using {torch.xpu.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No cuda or xpu, using cpu")

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


# tu paper goc

class PrioritizedReplayBuffer:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        buffer_size: int,
        mini_batch_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
    ):
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float32)
        self.action = torch.empty(buffer_size, dtype=torch.int64)
        self.reward = torch.empty(buffer_size, dtype=torch.float32)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float32)
        self.done = torch.empty(buffer_size, dtype=torch.int64)

        self.size = int(buffer_size)
        self.mini_batch_size = int(mini_batch_size)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

        self.count = 0
        self.real_size = 0

        self.sum_tree = _SumSegmentTree(self.size)
        self.min_tree = _MinSegmentTree(self.size)
        self.max_priority = 1.0  

    def __len__(self) -> int:
        return self.real_size

    def _priority(self, p: float) -> float:
        p = float(p)
        return float((abs(p) + self.eps) ** self.alpha)

    def add(self, transition):
        state, action, reward, next_state, done = transition

        self.state[self.count] = torch.as_tensor(state, dtype=torch.float32)
        self.action[self.count] = torch.as_tensor(action, dtype=torch.int64)
        self.reward[self.count] = torch.as_tensor(reward, dtype=torch.float32)
        self.next_state[self.count] = torch.as_tensor(next_state, dtype=torch.float32)
        self.done[self.count] = torch.as_tensor(done, dtype=torch.int64)

        p = self._priority(self.max_priority)
        self.sum_tree[self.count] = p
        self.min_tree[self.count] = p

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, beta: float | None = None):
        assert self.real_size >= self.mini_batch_size

        beta = self.beta if beta is None else float(beta)

        total_p = self.sum_tree.sum(0, self.real_size)
        assert total_p > 0.0

        segment = total_p / self.mini_batch_size
        idxs = np.empty(self.mini_batch_size, dtype=np.int64)
        priorities = np.empty(self.mini_batch_size, dtype=np.float32)

        for i in range(self.mini_batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(s)
            if idx >= self.real_size:
                idx = self.real_size - 1
            idxs[i] = idx
            priorities[i] = self.sum_tree[idx]

        probs = priorities / np.float32(total_p)
        min_p = self.min_tree.min(0, self.real_size) / np.float32(total_p)
        min_p = float(min_p) if min_p > 0 else 1e-12
        max_w = (self.real_size * min_p) ** (-beta)

        weights = (self.real_size * probs) ** (-beta)
        weights = weights / np.float32(max_w)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

        batch = (
            self.state[idxs].to(device),
            self.action[idxs].to(device),
            self.reward[idxs].to(device),
            self.next_state[idxs].to(device),
            self.done[idxs].to(device),
            idxs,            
        )
        return batch

    def update_priorities(self, idxs, priorities):
        idxs = np.asarray(idxs, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float32)
        assert idxs.shape[0] == priorities.shape[0]

        for idx, p in zip(idxs, priorities):
            idx = int(idx)
            if idx < 0 or idx >= self.real_size:
                continue
            pr = self._priority(float(p))
            self.sum_tree[idx] = pr
            self.min_tree[idx] = pr
            if pr > self.max_priority:
                self.max_priority = float(pr)

# class ReplayBufferCNN:
#     def __init__(self, state_shape, action_size, buffer_size, mini_batch_size, device):
#         state_shape = tuple(state_shape)
#         self.device = device 

#         self.state = torch.empty((buffer_size, *state_shape), dtype=torch.uint8, device="cpu")
#         self.action = torch.empty((buffer_size,), dtype=torch.int64, device="cpu")
#         self.reward = torch.empty((buffer_size,), dtype=torch.float32, device="cpu")
#         self.next_state = torch.empty((buffer_size, *state_shape), dtype=torch.uint8, device="cpu")
#         self.done = torch.empty((buffer_size,), dtype=torch.uint8, device="cpu")

#         self.count = 0
#         self.real_size = 0
#         self.size = int(buffer_size)
#         self.mini_batch_size = int(mini_batch_size)

#     def add(self, transition):
#         state, action, reward, next_state, done = transition
#         self.state[self.count] = torch.as_tensor(state, dtype=torch.uint8, device="cpu")
#         self.action[self.count] = torch.as_tensor(action, dtype=torch.int64, device="cpu")
#         self.reward[self.count] = torch.as_tensor(reward, dtype=torch.float32, device="cpu")
#         self.next_state[self.count] = torch.as_tensor(next_state, dtype=torch.uint8, device="cpu")
#         self.done[self.count] = torch.as_tensor(done, dtype=torch.uint8, device="cpu")
#         self.count = (self.count + 1) % self.size
#         self.real_size = min(self.size, self.real_size + 1)

#     def sample(self):
#         assert self.real_size >= self.mini_batch_size
#         sample_idxs = np.random.choice(self.real_size, self.mini_batch_size, replace=False)
#         states = self.state[sample_idxs].float().div(255.0).to(self.device)
#         next_states = self.next_state[sample_idxs].float().div(255.0).to(self.device)

#         actions = self.action[sample_idxs].to(self.device)
#         rewards = self.reward[sample_idxs].to(self.device)
#         dones = self.done[sample_idxs].float().to(self.device)

#         return states, actions, rewards, next_states, dones

class ReplayBufferCNN:
    def __init__(self, state_shape, action_size, buffer_size, mini_batch_size, device):
        state_shape = tuple(state_shape)
        self.device = device

        self.state = torch.empty((buffer_size, *state_shape), dtype=torch.uint8, device="cpu")
        self.action = torch.empty((buffer_size,), dtype=torch.int64, device="cpu")
        self.reward = torch.empty((buffer_size,), dtype=torch.float32, device="cpu")
        self.next_state = torch.empty((buffer_size, *state_shape), dtype=torch.uint8, device="cpu")
        self.done = torch.empty((buffer_size,), dtype=torch.uint8, device="cpu")

        self.count = 0
        self.real_size = 0
        self.size = int(buffer_size)
        self.mini_batch_size = int(mini_batch_size)

    def add(self, transition):
        state, action, reward, next_state, done = transition
        self.state[self.count] = torch.as_tensor(state, dtype=torch.uint8, device="cpu")
        self.action[self.count] = torch.as_tensor(action, dtype=torch.int64, device="cpu")
        self.reward[self.count] = torch.as_tensor(reward, dtype=torch.float32, device="cpu")
        self.next_state[self.count] = torch.as_tensor(next_state, dtype=torch.uint8, device="cpu")
        self.done[self.count] = torch.as_tensor(done, dtype=torch.uint8, device="cpu")
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)


    #do nothing
    def add_batch(self, states_u8, actions, rewards, next_states_u8, dones):
        states_u8 = np.asarray(states_u8, dtype=np.uint8)
        next_states_u8 = np.asarray(next_states_u8, dtype=np.uint8)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.uint8)

        n = int(actions.shape[0])
        if n <= 0:
            return

        end = self.count + n
        if end <= self.size:
            sl = slice(self.count, end)
            self.state[sl] = torch.as_tensor(states_u8, dtype=torch.uint8, device="cpu")
            self.action[sl] = torch.as_tensor(actions, dtype=torch.int64, device="cpu")
            self.reward[sl] = torch.as_tensor(rewards, dtype=torch.float32, device="cpu")
            self.next_state[sl] = torch.as_tensor(next_states_u8, dtype=torch.uint8, device="cpu")
            self.done[sl] = torch.as_tensor(dones, dtype=torch.uint8, device="cpu")
        else:
            n1 = self.size - self.count
            sl1 = slice(self.count, self.size)
            sl2 = slice(0, end % self.size)

            self.state[sl1] = torch.as_tensor(states_u8[:n1], dtype=torch.uint8, device="cpu")
            self.action[sl1] = torch.as_tensor(actions[:n1], dtype=torch.int64, device="cpu")
            self.reward[sl1] = torch.as_tensor(rewards[:n1], dtype=torch.float32, device="cpu")
            self.next_state[sl1] = torch.as_tensor(next_states_u8[:n1], dtype=torch.uint8, device="cpu")
            self.done[sl1] = torch.as_tensor(dones[:n1], dtype=torch.uint8, device="cpu")

            self.state[sl2] = torch.as_tensor(states_u8[n1:], dtype=torch.uint8, device="cpu")
            self.action[sl2] = torch.as_tensor(actions[n1:], dtype=torch.int64, device="cpu")
            self.reward[sl2] = torch.as_tensor(rewards[n1:], dtype=torch.float32, device="cpu")
            self.next_state[sl2] = torch.as_tensor(next_states_u8[n1:], dtype=torch.uint8, device="cpu")
            self.done[sl2] = torch.as_tensor(dones[n1:], dtype=torch.uint8, device="cpu")

        self.count = end % self.size
        self.real_size = min(self.size, self.real_size + n)

    def sample(self):
        assert self.real_size >= self.mini_batch_size
        sample_idxs = np.random.choice(self.real_size, self.mini_batch_size, replace=False)

        states = self.state[sample_idxs].float().div(255.0).to(self.device)
        next_states = self.next_state[sample_idxs].float().div(255.0).to(self.device)

        actions = self.action[sample_idxs].to(self.device)
        rewards = self.reward[sample_idxs].to(self.device)
        dones = self.done[sample_idxs].float().to(self.device)

        return states, actions, rewards, next_states, dones