import torch
import torch.optim as optim
import numpy as np
from torch import nn
from utils import * 
import time
# device = ''

# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print(f"Using {torch.cuda.get_device_name(0)}")

# elif torch.xpu.is_available():
#     device = torch.device("xpu")
#     print(f"Using {torch.xpu.get_device_name(0)}")
# else:
#     device = torch.device("cpu")
#     print("No cuda or xpu, using cpu")

class PPO(nn.Module):
    def __init__(self, clip_coef, vf_coef, ent_coef, lr, max_grad_norm, update_epochs, num_minibatches,state_dim,action_dim,layer_size):
        # self.agent = agent
        super().__init__()
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.layer_size = layer_size

        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, layer_size)),
            nn.ReLU(),
            layer_init(nn.Linear(layer_size,layer_size )),
            nn.ReLU(),
            layer_init(nn.Linear(layer_size, 1), std=1.0) 
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, layer_size)),
            nn.ReLU(),
            layer_init(nn.Linear(layer_size, layer_size)),
            nn.ReLU(),
            layer_init(nn.Linear(layer_size, action_dim), std=0.01) 
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, state_value = self.policy_old.act(state_tensor)
        return action, log_prob, state_value.flatten(), state_tensor
    
    def get_value(self, x):
        return self.critic(x).flatten()

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(x).flatten()
        return action, log_prob, entropy, value

    def update(self, buffer):
        model_device = next(self.parameters()).device  
        t0 = time.time()


        batch = buffer.get()
        b_obs = batch['obs'].to(model_device)
        b_logprobs = batch['logprobs'].to(model_device)
        b_actions = batch['actions'].to(model_device)
        b_advantages = batch['advantages'].to(model_device)
        b_returns = batch['returns'].to(model_device)
        b_values = batch['values'].to(model_device)

        batch_size = b_obs.shape[0]
        minibatch_size = batch_size // self.num_minibatches
        b_inds = np.arange(batch_size)

        # if model_device.type == "xpu" and hasattr(torch, "xpu"):
        #     torch.xpu.synchronize()
        # print(f"[PPO.update] move batch done: {time.time()-t0:.3f}s", flush=True)

        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]


                t1 = time.time()


                _, newlogprob, entropy, newvalue = self.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )


                # if model_device.type == "xpu" and hasattr(torch, "xpu"):
                #     torch.xpu.synchronize()
                # print(f"[PPO.update] forward done: {time.time()-t1:.3f}s", flush=True)


                t2 = time.time()


                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -self.clip_coef, self.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()

                foreach = False if next(self.parameters()).device.type == "xpu" else True

                
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm,foreach=foreach)
                self.optimizer.step()


                # if model_device.type == "xpu" and hasattr(torch, "xpu"):
                #     torch.xpu.synchronize()
                # print(f"[PPO.update] backward/step done: {time.time()-t2:.3f}s", flush=True)


                last_loss = float(loss.detach().cpu())

        return last_loss
