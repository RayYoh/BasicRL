import numpy as np
from typing import Dict, List, Tuple, Deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class OffPolicyBuffer():
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        
        batch_tensor = {}
        for k, v in batch.items():
            if k == 'obs':
                batch_tensor[k] = torch.as_tensor(v, dtype=torch.float32)
            if k == 'obs2':
                batch_tensor[k] = torch.as_tensor(v, dtype=torch.float32)
            if k == 'act':
                batch_tensor[k] = torch.as_tensor(v, dtype=torch.int64).reshape(-1,1)
            if k == 'rew':
                batch_tensor[k] = torch.as_tensor(v, dtype=torch.float32).reshape(-1,1)
            if k == 'done':
                batch_tensor[k] = torch.as_tensor(v, dtype=torch.float32).reshape(-1,1)
        return batch_tensor

class Network(nn.Module):
    """
    Network architecture:  

    Network(
    (layers): Sequential(
        (0): Linear(in_features=in_dim, out_features=hidden_size, bias=True)
        (1): ReLU()
        (2): Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        (3): ReLU()
        (4): Linear(in_features=hidden_size, out_features=out_dim, bias=True)
        )
    )
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int = 128):
        """Initialization of the Network."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)

class DuelingNetwork(nn.Module):
    """
    DuelingNetwork(
        (feature_layer): Sequential(
            (0): Linear(in_features=in_dim, out_features=hidden_size, bias=True)
            (1): ReLU()
        )
        (advantage_layer): Sequential(
            (0): Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
            (1): ReLU()
            (2): Linear(in_features=12hidden_size8, out_features=2, bias=True)
        )
        (value_layer): Sequential(
            (0): Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
            (1): ReLU()
            (2): Linear(in_features=hidden_size, out_features=1, bias=True)
        )
    )
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int = 128):
        """Initialization of the Network."""
        super(DuelingNetwork, self).__init__()
        #common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
        )

        #advantege layer
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
        )

        #value layer
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(x)

        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

class NoisyNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int = 128):
        """Initialization."""
        super(NoisyNetwork, self).__init__()

        self.feature = nn.Linear(in_dim, hidden_size)
        self.noisy_layer1 = NoisyLinear(hidden_size, hidden_size)
        self.noisy_layer2 = NoisyLinear(hidden_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = F.relu(self.feature(x))
        hidden = F.relu(self.noisy_layer1(feature))
        out = self.noisy_layer2(hidden)
        
        return out
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()

class DistributionalNetwork(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor,
        hidden_size: int = 128,
    ):
        """Initialization."""
        super(DistributionalNetwork, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, out_dim * atom_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size) # Reconstructing the dimension of the tensor.
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist