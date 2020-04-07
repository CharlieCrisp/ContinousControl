import numpy as np


class PPOAgent:

    def __init__(self, num_agents, action_size):
        self.action_size = action_size
        self.num_agents = num_agents

    def act(self, state, train=True):
        actions = np.random.randn(self.num_agents, self.action_size)
        actions = np.clip(actions, -1, 1)
        return actions

    def learn(self, states, actions, rewards, next_states, dones):
        return

    def save_agent_state(self, output_file):
        print("Saving agent state")
