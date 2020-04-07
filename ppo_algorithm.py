import numpy as np

from unityagents import UnityEnvironment

from ppo_agent import PPOAgent
from solver import Solver


max_t = 100


def _collect_trajectory(agent: PPOAgent, env: UnityEnvironment, brain_name: str):
    trajectory_states = []
    trajectory_actions = []
    trajectory_rewards = []
    trajectory_next_states = []
    trajectory_dones = []

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
        trajectory_next_states.append(next_states)
        trajectory_dones.append(dones)

        state = next_states
        if np.any(dones):
            break

    return trajectory_states, trajectory_actions, trajectory_rewards, trajectory_next_states, trajectory_dones


def ppo(agent: PPOAgent, env: UnityEnvironment, brain_name: str, n_rollouts: int, batch_size: int, solver: Solver,
        solved_agent_output_file: str):
    trajectories = []
    for i in range(n_rollouts):
        trajectory = _collect_trajectory(agent, env, brain_name)
        trajectories.append(trajectory)
        solver.record_trajectory(trajectory)
        if i % batch_size:
            states, actions, rewards, next_states, dones = list(zip(*trajectories))
            agent.learn(states, actions, rewards, next_states, dones)
            trajectories.clear()

        if solver.is_solved():
            print(f"Solved the environment in ${i} rollouts")
            agent.save_agent_state(solved_agent_output_file)
            return
