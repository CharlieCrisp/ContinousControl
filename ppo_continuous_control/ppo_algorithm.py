from typing import Tuple, List, Union

import numpy as np
import torch

from unityagents import UnityEnvironment, AllBrainInfo

from ppo_continuous_control.ppo_agent import PPOAgent
from ppo_continuous_control.progress_tracker import ProgressTracker
from ppo_continuous_control.solver import Solver

max_t = 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_future_rewards(rewards: np.ndarray, gamma=0.99) -> torch.Tensor:
    n = len(rewards)
    gammas = gamma ** np.arange(n)
    gammas_matrix = [
        np.concatenate((np.zeros(i), gammas[: n - i])) for i in range(len(rewards))
    ]

    rewards = np.expand_dims(rewards, 1)

    return torch.tensor(np.matmul(gammas_matrix, rewards).squeeze())


def _collect_trajectory(
    agent: PPOAgent, env: UnityEnvironment, brain_name: str, env_info
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, AllBrainInfo]:
    trajectory_states = []
    trajectory_actions = []
    trajectory_rewards = []

    state = torch.tensor(env_info.vector_observations).to(device).float()
    for i in range(max_t):
        actions = agent.act(state)
        env_info = env.step(actions)[brain_name]

        next_states = torch.tensor(env_info.vector_observations).to(device).float()
        rewards = env_info.rewards
        dones = env_info.local_done
        trajectory_states.append(state)
        trajectory_actions.append(torch.tensor(actions).float())
        trajectory_rewards.append(rewards)

        state = next_states
        if np.any(dones):
            break

    return (
        torch.cat(trajectory_states),
        torch.cat(trajectory_actions),
        np.concatenate(trajectory_rewards),
        env_info,
    )


def record_rewards(new_future_rewards, new_rewards, progress_trackers, solver):
    solver.record_rewards(new_rewards)
    score = new_future_rewards[0]
    if isinstance(progress_trackers, list):
        for tracker in progress_trackers:
            tracker.record_score(score)
    elif progress_trackers is not None:
        progress_trackers.record_score(score)


def ppo(
    agent: PPOAgent,
    env: UnityEnvironment,
    brain_name: str,
    n_rollouts: int,
    batch_size: int,
    solver: Solver,
    solved_agent_output_file: str,
    progress_trackers: Union[List[ProgressTracker], ProgressTracker] = None,
):
    states, actions, future_rewards = [], [], []
    env_info = env.reset(train_mode=True)[brain_name]

    for i in range(n_rollouts):
        new_states, new_actions, new_rewards, env_info = _collect_trajectory(
            agent, env, brain_name, env_info
        )
        new_future_rewards = calculate_future_rewards(new_rewards).to(device)
        states.append(new_states)
        actions.append(new_actions)
        future_rewards.append(new_future_rewards)

        record_rewards(new_future_rewards, new_rewards, progress_trackers, solver)

        if i % batch_size:
            rewards = torch.cat(future_rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            agent.learn(
                torch.cat(states), torch.cat(actions), rewards
            )
            states.clear()
            actions.clear()
            future_rewards.clear()

        if solver.is_solved():
            print(f"Solved the environment in ${i} rollouts")
            agent.save_agent_state(solved_agent_output_file)
            return
