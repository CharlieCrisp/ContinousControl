import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from ppo_continuous_control.action_taker import ActionTaker
from ppo_continuous_control.policy_nn import PolicyNetwork


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent:
    #  TODO use actor critic
    def __init__(
        self,
        num_agents: int,
        state_size: int,
        action_size: int,
        action_taker: ActionTaker,
        epsilon=0.2,
        learning_rate=0.0003,
    ):
        self.action_size = action_size
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.action_taker = action_taker
        self.model: nn.Module = PolicyNetwork(state_size, action_size, seed=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action_means = self.model(state).detach()
            return self.action_taker.take_action(action_means).data.numpy()

    def _probabilities_of_actions(
        self, states: torch.Tensor, actions: torch.Tensor, detach=False
    ) -> torch.FloatTensor:
        """
        :param detach: When true, the calculation will be done detached and therefore will not affect auto-diff.
                       The output will be treated as a constant when gradients are calculated.
        :return: The probabilities of taking the given actions from the given states
        """

        if detach:
            with torch.no_grad():
                action_means = self.model(states).detach()
        else:
            action_means = self.model(states)

        return self.action_taker.probability_of_action(
            nn_output=action_means, actions=actions
        )

    def learn(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        future_rewards,
        num_learning_iterations=80,
    ):
        starting_probabilities = self._probabilities_of_actions(
            states, actions, detach=True
        )

        for learning_iteration in range(num_learning_iterations):
            # TODO check that these all have the correct shape
            new_probabilities = self._probabilities_of_actions(states, actions)
            probability_ratio = new_probabilities / starting_probabilities
            clipped_probability_ratio = probability_ratio.clamp(
                min=1 - self.epsilon, max=1 + self.epsilon
            )
            surrogate = -clipped_probability_ratio * future_rewards

            # TODO include entropy in surrogate

            self.optimizer.zero_grad()
            surrogate.mean().backward()
            self.optimizer.step()

    def save_agent_state(self, output_file):
        torch.save(self.model.state_dict(), output_file)
