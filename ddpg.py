import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class OANoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, X0=None):
        self.theta = theta
        self.sigma = sigma
        self.X0 = X0
        self.dt = dt
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.max_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))

        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random(choice(max_mem, batch_size))

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        reward = self.reward_memory[batch]
        action = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, action, reward, new_states, terminal


class Critic(nn.Module):
    def __Init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='/tmp/ddpg'):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc2_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.chkpt_dir = os.path.join(chkpt_dir, name + '_ddpg')
        self.name = name

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.wheth.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)

        torch.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.bn1 = nn.LeayerNormal(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2_weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)

        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)

        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value




