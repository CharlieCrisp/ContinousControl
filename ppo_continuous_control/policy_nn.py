from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        super(ActorNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        layer1_size = 256
        self.layer1 = nn.Linear(state_size, layer1_size)
        layer2_size = 128
        self.means_layer3 = nn.Linear(layer1_size, layer2_size)
        self.stds_layer3 = nn.Linear(layer1_size, layer2_size)

        self.means_output = nn.Linear(layer2_size, action_size)
        self.stds_output = nn.Linear(layer2_size, action_size)

    def forward(self, state):
        x = torch.tanh(self.layer1(state))
        means = torch.tanh(self.means_layer3(x))
        means = torch.tanh(self.means_output(means))
        stds = torch.tanh(self.stds_layer3(x))
        stds = torch.tanh(self.stds_output(stds))
        return means, stds


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.actor = ActorNN(state_size, action_size, seed)

    def __get_action_distribution(self, action_means, action_stds):
        action_variances = action_stds ** 2 + 1e-5
        cov_mat = torch.diag_embed(action_variances).to(device)
        return MultivariateNormal(loc=action_means, covariance_matrix=cov_mat)

    def act(self, state):
        action_means, action_stds = self.actor(state)
        action_distribution = self.__get_action_distribution(action_means, action_stds)
        return action_distribution.sample().detach()

    def log_probability_and_entropy_of_action(self, states, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        action_means, action_stds = self.actor(states)
        action_distribution = self.__get_action_distribution(action_means, action_stds)

        return (
            action_distribution.log_prob(actions).double(),
            action_distribution.entropy()
        )

    def forward(self, state):
        raise NotImplementedError("This function should not be called")


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze()
