import torch
from torch import nn
import torch.nn.functional as F
# from utils import *

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, enable_dueling_dqn=True):
        super().__init__()
        self.enable_dueling_dqn = bool(enable_dueling_dqn)

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        if self.enable_dueling_dqn:
            self.v_fc = nn.Linear(hidden_dim, hidden_dim)
            self.v_ln = nn.LayerNorm(hidden_dim)
            self.v_out = nn.Linear(hidden_dim, 1)

            self.a_fc = nn.Linear(hidden_dim, hidden_dim)
            self.a_ln = nn.LayerNorm(hidden_dim)
            self.a_out = nn.Linear(hidden_dim, action_dim)
        else:
            self.q_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept single state (state_dim,) -> (1, state_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))

        if self.enable_dueling_dqn:
            v = F.relu(self.v_ln(self.v_fc(x)))
            V = self.v_out(v)  # (B,1)

            a = F.relu(self.a_ln(self.a_fc(x)))
            A = self.a_out(a)  # (B,action_dim)

            Q = V + A - A.mean(dim=1, keepdim=True)
        else:
            Q = self.q_out(x)

        return Q




if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim,128,False)
    state = torch.tensor([ 1.0000,  0.2344,  0.4297,  1.0000,  0.0000,  1.0000,  1.0000,  0.0000,
          1.0000,  0.4766, -0.9000,  0.5000])
    with torch.no_grad():
        output = net(state)
        print(output)
        print(output.argmax())
    # print(output)

