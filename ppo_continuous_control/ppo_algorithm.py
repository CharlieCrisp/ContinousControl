from typing import Tuple, List, Union

import numpy as np
import torch

from unityagents import UnityEnvironment, AllBrainInfo

from ppo_continuous_control.ppo_agent import PPOAgent
from ppo_continuous_control.progress_tracker import ProgressTracker
from ppo_continuous_control.solver import Solver

max_t = 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_future_rewards(
    rewards: np.ndarray, dones: np.ndarray, gamma=0.99
) -> torch.Tensor:
    discounted_reward = 0
    discounted_rewards = []
    for reward, done in zip(np.flip(rewards), np.flip(dones)):
        discounted_reward = reward + discounted_reward * gamma if not done else reward
        discounted_rewards.append(discounted_reward)

    return torch.flip(torch.tensor(discounted_rewards, dtype=torch.float64), dims=(0,))


def _collect_trajectory(
    agent: PPOAgent, env: UnityEnvironment, brain_name: str, num_agents
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    trajectory_states = []
    trajectory_actions = []
    trajectory_rewards = []

    env_info = env.reset(train_mode=True)[brain_name]
    state = torch.tensor(env_info.vector_observations).to(device).float()
    for i in range(max_t):
        actions = agent.act(state).data.cpu().numpy()
        trajectory_states.append(state)
        trajectory_actions.append(torch.tensor(actions).float())

        env_info = env.step(actions)[brain_name]
        trajectory_rewards.append(env_info.rewards)

        state = torch.tensor(env_info.vector_observations).to(device).float()
        dones = env_info.local_done

        if np.any(dones):
            break

    num_steps = len(trajectory_states)
    single_agent_dones = np.full((num_steps,), False, dtype=bool)
    single_agent_dones[num_steps - 1] = True

    all_agent_dones = np.tile(single_agent_dones, num_agents).reshape(
        num_agents, num_steps
    )

    states = torch.stack(trajectory_states).permute(dims=[1, 0, 2])
    actions = torch.stack(trajectory_actions).permute(dims=[1, 0, 2])
    rewards = np.swapaxes(np.array(trajectory_rewards), 1, 0)

    return (
        states,
        actions,
        rewards,
        all_agent_dones,
    )


def record_rewards(new_rewards, progress_trackers, solver):
    solver.record_rewards(new_rewards)
    score = new_rewards.sum() / 20
    if isinstance(progress_trackers, list):
        for tracker in progress_trackers:
            tracker.record_score(score)
    elif progress_trackers is not None:
        progress_trackers.record_score(score)


def ppo(
    agent: PPOAgent,
    num_agents: int,
    state_size,
    action_size,
    env: UnityEnvironment,
    brain_name: str,
    n_rollouts: int,
    batch_size: int,
    solver: Solver,
    progress_trackers: Union[List[ProgressTracker], ProgressTracker] = None,
):
    states, actions, future_rewards = [], [], []

    for i in range(n_rollouts):
        new_states, new_actions, new_rewards, new_dones = _collect_trajectory(
            agent, env, brain_name, num_agents
        )
        len_trajectory = new_states.size()[1]
        new_states = new_states.reshape([num_agents * len_trajectory, state_size])
        new_actions = new_actions.reshape([num_agents * len_trajectory, action_size])
        new_rewards = new_rewards.reshape([num_agents * len_trajectory])
        new_dones = new_dones.reshape([num_agents * len_trajectory])

        new_future_rewards = calculate_future_rewards(new_rewards, new_dones).to(device)

        states.append(new_states)
        actions.append(new_actions)
        future_rewards.append(new_future_rewards)
        record_rewards(new_rewards, progress_trackers, solver)

        if i != 0 and i % batch_size == 0:
            agent.learn(
                torch.cat(states), torch.cat(actions), torch.cat(future_rewards)
            )
            states.clear()
            actions.clear()
            future_rewards.clear()

        if solver.is_solved():
            print(f"Solved the environment in {i} rollouts")
            agent.save_agent_state()
