import os
from argparse import ArgumentParser

from unityagents import UnityEnvironment

from ppo_continuous_control.ddpg_algorithm import ddpg
from ppo_continuous_control.ppo_agent import PPOAgent
from ppo_continuous_control.ppo_algorithm import ppo
from ppo_continuous_control.progress_tracker import (
    ScoreGraphPlotter,
    ProgressBarTracker,
)
from ppo_continuous_control.solver import AverageScoreSolver


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main(output_file, n_rollouts, use_multiple_agents, algorithm, saved_weights):
    print("Creating Unity environment for Reacher app")
    env = (
        UnityEnvironment(file_name="ReacherMultiple.app")
        if use_multiple_agents
        else UnityEnvironment(file_name="Reacher.app")
    )

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    brain = env.brains[brain_name]

    print("Initialising PPO Agent")
    state_size = brain.vector_observation_space_size
    action_size = brain.vector_action_space_size
    print(f"Using state size {state_size} and action size {action_size}")

    num_agents = len(env_info.agents)
    agent = PPOAgent(num_agents, state_size, action_size, saved_weights)
    batch_size = 1
    solver = AverageScoreSolver(
        solved_score=30, solved_score_period=100, num_agents=num_agents
    )
    plotter = ScoreGraphPlotter(
        score_min=0, score_max=12, solved_score=30, solved_score_period=100
    )
    progress_bar = ProgressBarTracker(n_rollouts)

    if algorithm == "ppo":
        print("Running PPO algorithm")
        ppo(
            agent,
            num_agents,
            state_size,
            action_size,
            env,
            brain_name,
            n_rollouts,
            batch_size,
            solver,
            output_file,
            [plotter, progress_bar],
        )
    elif algorithm == "ddpg":
        print("Running DDPG algorithm")
        ddpg(
            agent,
            num_agents,
            state_size,
            action_size,
            env,
            brain_name,
            n_rollouts,
            batch_size,
            solver,
            output_file,
            [plotter, progress_bar],
        )
    else:
        print(f"Algorithm '{algorithm}' is not supported")

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
    args_parser.add_argument(
        "--use-multiple-agents",
        dest="use_multiple_agents",
        help="Pass this in to train agent using environment with 20 agents acting simultaneously.",
        action="store_true",
    )
    args_parser.add_argument(
        "--algo",
        dest="algorithm",
        help="The algorithm to use to train agent. Options: ppo, ddpg",
        default="ppo",
    )
    args_parser.add_argument(
        "--actor-weights",
        dest="actor_weights",
        help="The filename of saved weights to use as the starting state of an agents actor. "
        + "Note: you must also pass in --critic-weights",
        default=None,
    )

    args_parser.add_argument(
        "--critic-weights",
        dest="critic_weights",
        help="The filename of saved weights to use as the starting state of an agents critic. "
        + "Note: you must also pass in --actor-weights",
        default=None,
    )
    args = args_parser.parse_args()

    saved_weights = (
        (args.actor_weights, args.critic_weights)
        if args.actor_weights is not None and args.critic_weights is not None
        else None
    )

    main(
        args.filename,
        args.n_rollouts,
        args.use_multiple_agents,
        args.algorithm,
        saved_weights,
    )
