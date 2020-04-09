from typing import Tuple

import numpy as np
import torch

from unityagents import UnityEnvironment

from ppo_continuous_control.ppo_agent import PPOAgent
from ppo_continuous_control.solver import Solver

max_t = 100


def calculate_future_rewards(rewards: np.ndarray) -> np.ndarray:
    cumulative_score = rewards.cumsum()
    final_score = cumulative_score[cumulative_score.size - 1]
    return final_score - cumulative_score + rewards


def _collect_trajectory(
    agent: PPOAgent, env: UnityEnvironment, brain_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    trajectory_states = []
    trajectory_actions = []
    trajectory_rewards = []

    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations
    for i in range(max_t):
        actions = agent.act(state)
        env_info = env.step(actions)[brain_name]

        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        trajectory_states.append(state)
        trajectory_actions.append(actions)
        trajectory_rewards.append(rewards)

        state = next_states
        if np.any(dones):
            print("Finished episode")
            break

    return (
        np.array(trajectory_states),
        np.array(trajectory_actions),
        np.array(trajectory_rewards),
    )


def ppo(
    agent: PPOAgent,
    env: UnityEnvironment,
    brain_name: str,
    n_rollouts: int,
    batch_size: int,
    solver: Solver,
    solved_agent_output_file: str,
):
    states, actions, future_rewards = [], [], []

    for i in range(n_rollouts):
        new_states, new_actions, new_rewards = _collect_trajectory(agent, env, brain_name)
        new_future_rewards = torch.tensor(calculate_future_rewards(new_rewards)).float()
        states.append(new_states)
        actions.append(new_actions)
        future_rewards.append(new_future_rewards)

        solver.record_rewards(new_rewards)

        if i % batch_size:
            agent.learn(states, actions, future_rewards)
            states.clear()
            actions.clear()
            future_rewards.clear()

        if solver.is_solved():
            print(f"Solved the environment in ${i} rollouts")
            agent.save_agent_state(solved_agent_output_file)
            return
