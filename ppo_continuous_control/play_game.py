import os
import numpy as np
import torch

from argparse import ArgumentParser
from unityagents import UnityEnvironment

from ppo_continuous_control.ppo_agent import PPOAgent

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(n_rollouts, use_multiple_agents, saved_weights):
    print("Creating Unity environment for Reacher app")
    env = (
        UnityEnvironment(file_name="ReacherMultiple.app")
        if use_multiple_agents
        else UnityEnvironment(file_name="Reacher.app")
    )

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    brain = env.brains[brain_name]

    state_size = brain.vector_observation_space_size
    action_size = brain.vector_action_space_size
    print(f"Using state size {state_size} and action size {action_size}")

    num_agents = len(env_info.agents)
    agent = PPOAgent(num_agents, state_size, action_size, saved_weights=saved_weights)

    for i in range(n_rollouts):
        env_info = env.reset(train_mode=True)[brain_name]
        state = torch.tensor(env_info.vector_observations).to(device).float()
        for j in range(1000):
            actions = agent.act(state).data.cpu().numpy()
            env_info = env.step(actions)[brain_name]
            state = torch.tensor(env_info.vector_observations).to(device).float()
            if np.any(env_info.local_done):
                break

    print("Closing environment")
    env.close()


if __name__ == "__main__":
    args_parser = ArgumentParser(
        description="A script to run forward the continuous control environment using a trained agent"
    )
    args_parser.add_argument(
        "-n",
        type=int,
        dest="n_rollouts",
        help="The number of trajectories to collect whilst training",
        default=1000,
    )
    args_parser.add_argument(
        "--use-multiple-agents",
        dest="use_multiple_agents",
        help="Pass this in to train agent using environment with 20 agents acting simultaneously.",
        action="store_true",
    )
    args_parser.add_argument(
        "--actor-weights",
        dest="actor_weights",
        help="The filename of saved weights to use as the starting state of an agents actor. "
        + "Note: you must also pass in --critic-weights",
        default="saved_actor.pth",
    )

    args_parser.add_argument(
        "--critic-weights",
        dest="critic_weights",
        help="The filename of saved weights to use as the starting state of an agents critic. "
        + "Note: you must also pass in --actor-weights",
        default="saved_critic.pth",
    )
    args = args_parser.parse_args()

    saved_weights = (
        (args.actor_weights, args.critic_weights)
        if args.actor_weights is not None and args.critic_weights is not None
        else None
    )

    main(
        args.n_rollouts,
        args.use_multiple_agents,
        saved_weights,
    )
