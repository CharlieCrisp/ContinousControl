import os
from argparse import ArgumentParser

import torch
from unityagents import UnityEnvironment

from ppo_continuous_control.action_taker import NormalDistributionActionTaker
from ppo_continuous_control.ppo_agent import PPOAgent
from ppo_continuous_control.ppo_algorithm import ppo
from ppo_continuous_control.progress_tracker import (
    ScoreGraphPlotter,
    ProgressBarTracker,
)
from ppo_continuous_control.solver import NeverSolved


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main(output_file, n_rollouts):
    print("Creating Unity environment for Reacher app")
    env = UnityEnvironment(file_name="Reacher.app")

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    brain = env.brains[brain_name]

    print("Initialising PPO Agent")
    state_size = brain.vector_observation_space_size
    action_size = brain.vector_action_space_size
    print(f"Using state size {state_size} and action size {action_size}")

    action_standard_deviation = 0.5
    action_taker = NormalDistributionActionTaker(
        torch.full(
            size=(action_size,),
            fill_value=(action_standard_deviation * action_standard_deviation),
        )
    )
    agent = PPOAgent(len(env_info.agents), state_size, action_size, action_taker)
    batch_size = 4
    solver = NeverSolved()
    plotter = ScoreGraphPlotter(score_min=-1, score_max=12)
    progress_bar = ProgressBarTracker(n_rollouts)

    print("Running PPO algorithm")
    ppo(
        agent,
        env,
        brain_name,
        n_rollouts,
        batch_size,
        solver,
        output_file,
        [plotter, progress_bar],
    )

    print("Finished running PPO. Closing environment")
    env.close()


if __name__ == "__main__":
    args_parser = ArgumentParser(
        description="A script to train and run an agent in the continuous control environment"
    )
    args_parser.add_argument(
        "--output-file",
        type=str,
        dest="filename",
        help="The file to save the trained agent weights to",
        default="trained_agent_weights.pth",
    )
    args_parser.add_argument(
        "--n",
        type=str,
        dest="n_rollouts",
        help="The number of trajectories to collect whilst training",
        default=1000,
    )
    args = args_parser.parse_args()

    main(args.filename, args.n_rollouts)
