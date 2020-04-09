from argparse import ArgumentParser
from unityagents import UnityEnvironment

from ppo_continuous_control.ppo_agent import PPOAgent
from ppo_continuous_control.ppo_algorithm import ppo
from ppo_continuous_control.solver import Solver


def main(output_file, n_rollouts):
    print("Creating Unity environment for Reacher app")
    env = UnityEnvironment(file_name="Reacher.app")

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    brain = env.brains[brain_name]

    print("Initialising PPO Agent")
    agent = PPOAgent(len(env_info.agents), brain.vector_observation_space_size, brain.vector_action_space_size)
    batch_size = 20
    solver = Solver()

    print("Running PPO algorithm")
    ppo(agent, env, brain_name, n_rollouts, batch_size, solver, output_file)

    print("Finished running PPO. Closing environment")
    env.close()


if __name__ == "__main__":
    args_parser = ArgumentParser(description="A script to train and run an agent in the continuous control environment")
    args_parser.add_argument(
        "--output-file",
        type=str,
        dest="filename",
        help="The file to save the trained agent weights to",
        default="trained_agent_weights.pth",
    )
    args_parser.add_argument(
        "--n", type=str, dest="n_rollouts", help="The number of trajectories to collect whilst training", default=1000,
    )
    args = args_parser.parse_args()

    main(args.filename, args.n_rollouts)
