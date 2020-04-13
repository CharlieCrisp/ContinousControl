from typing import Tuple, List, Union

import numpy as np
import torch

from unityagents import UnityEnvironment, AllBrainInfo

from ppo_continuous_control.ppo_agent import PPOAgent
from ppo_continuous_control.progress_tracker import ProgressTracker
from ppo_continuous_control.solver import Solver

max_t = 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ddpg(
    agent: PPOAgent,
    num_agents: int,
    state_size,
    action_size,
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
        new_states, new_actions, new_rewards, new_dones, env_info = _collect_trajectory(
            agent, env, brain_name, env_info, num_agents
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

        record_rewards(new_future_rewards, new_rewards, progress_trackers, solver)

        if i != 0 and i % batch_size == 0:
            agent.learn(
                torch.cat(states), torch.cat(actions), torch.cat(future_rewards)
            )
            states.clear()
            actions.clear()
            future_rewards.clear()

        if solver.is_solved():
            print(f"Solved the environment in ${i} rollouts")
            agent.save_agent_state(solved_agent_output_file)
            return
