import numpy as np


def calculate_future_rewards(rewards: np.ndarray):
    cumulative_score = rewards.cumsum()
    final_score = cumulative_score[cumulative_score.size - 1]
    return final_score - cumulative_score + rewards


class PPOAgent:

    def __init__(self, num_agents, action_size):
        self.action_size = action_size
        self.num_agents = num_agents

    def act(self, state, train=True):
        actions = np.random.randn(self.num_agents, self.action_size)
        actions = np.clip(actions, -1, 1)
        return actions

    def learn(self, states, actions, rewards, next_states, dones):
        future_rewards = calculate_future_rewards(rewards)
        return

    def save_agent_state(self, output_file):
        print("Saving agent state")
