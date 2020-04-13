from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import mse_loss

from ppo_continuous_control.policy_nn import Critic, Actor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent:
    def __init__(
        self,
        num_agents: int,
        state_size: int,
        action_size: int,
        epsilon=0.2,
        learning_rate=0.0003,
    ):
        self.action_size = action_size
        self.num_agents = num_agents
        self.epsilon = epsilon

        self.actor = Actor(state_size, action_size, seed=0).to(device)
        self.old_actor: nn.Module = Actor(state_size, action_size, seed=0).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def act(self, state: torch.Tensor) -> np.ndarray:
        return self.actor.act(state).detach()

    def _log_probabilities_and_entropies_of_actions(
        self, states: torch.Tensor, actions: torch.Tensor, detach=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param detach: When true, the calculation will be done detached and therefore will not affect auto-diff.
                       The output will be treated as a constant when gradients are calculated.
        :return: The probabilities of taking the given actions from the given states
        """
        probs, entropies = self.actor.log_probability_and_entropy_of_action(states, actions)

        if detach:
            probs = probs.detach()
            entropies = entropies.detach()
        return probs, entropies

    def learn(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        future_rewards,
        num_learning_iterations=30,
    ):
        actions = actions.detach()
        states = states.detach()
        future_rewards = future_rewards.detach()

        starting_log_probabilities, _ = self._log_probabilities_and_entropies_of_actions(
            states, actions, detach=True
        )

        # Train critic
        state_values = self.critic(states).double()

        critic_loss = mse_loss(state_values, future_rewards)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train actor
        for i in range(num_learning_iterations):
            (
                new_log_probabilities,
                new_entropies
            ) = self._log_probabilities_and_entropies_of_actions(states, actions)
            probability_ratio = torch.exp(new_log_probabilities - starting_log_probabilities.detach())

            advantages = future_rewards - state_values.detach()
            clipped_probability_ratio = probability_ratio.clamp(
                min=1 - self.epsilon, max=1 + self.epsilon
            )
            surr1 = clipped_probability_ratio * advantages
            surr2 = probability_ratio * advantages

            loss = -(torch.min(surr1, surr2) + 0.01 * new_entropies)

            self.actor_optimizer.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()
        self.old_actor.load_state_dict(self.actor.state_dict())

    def save_agent_state(self, output_file):
        torch.save(self.actor.state_dict(), output_file)
