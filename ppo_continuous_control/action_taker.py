import abc
from abc import abstractmethod

import torch
from torch.distributions import MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActionTaker(abc.ABC):
    @abstractmethod
    def take_action(self, nn_output) -> torch.Tensor:
        pass

    @abstractmethod
    def probability_of_action(self, nn_output, actions):
        pass


class NormalDistributionActionTaker(ActionTaker):
    def __init__(self, action_variances, minimum=-1, maximum=1):
        self.action_variances = action_variances
        self.min = minimum
        self.max = maximum

    def __get_action_distribution(self, action_means):
        return MultivariateNormal(
            loc=action_means,
            covariance_matrix=torch.diag(self.action_variances).to(device),
        )

    def take_action(self, nn_output):
        action_distribution = self.__get_action_distribution(action_means=nn_output)
        return action_distribution.sample().clamp(min=self.min, max=self.max)

    def probability_of_action(self, nn_output, actions):
        action_distribution = self.__get_action_distribution(nn_output)
        return torch.exp(action_distribution.log_prob(actions))
