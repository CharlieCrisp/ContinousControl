import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from ppo_continuous_control.policy_nn import PolicyNetwork


class PPOAgent:
    def __init__(self, num_agents, state_size, action_size, epsilon=0.1, learning_rate=5e-4):
        self.action_size = action_size
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.model: nn.Module = PolicyNetwork(state_size, action_size, seed=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, state, train=True):
        #  TODO
        actions = np.random.randn(self.num_agents, self.action_size)
        actions = np.clip(actions, -1, 1)
        return actions

    def _probabilities_of_actions(self, states, actions, detach=False) -> torch.FloatTensor:
        """
        :param detach: When true, the calculation will be done detached and therefore will not affect auto-diff.
                       The output will be treated as a constant when gradients are calculated.
        :return: The probabilities of taking the given actions from the given states
        """
        return self.model(states).gather(actions)  # TODO: Fix this pseudo code

    def learn(self, states, actions, future_rewards, num_learning_iterations=3):
        starting_probabilities = self._probabilities_of_actions(states, actions, detach=True)

        for learning_iteration in range(num_learning_iterations):
            new_probabilities = self._probabilities_of_actions(states, actions)
            probability_ratio = new_probabilities / starting_probabilities
            clipped_probability_ratio = probability_ratio.clamp(max=1 + self.epsilon)
            surrogate = -(clipped_probability_ratio * future_rewards).sum()

            self.optimizer.zero_grad()
            surrogate.backward()
            self.optimizer.step()

    def save_agent_state(self, output_file):
        torch.save(self.model.state_dict(), output_file)
