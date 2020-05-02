# Continuous Control
This is a submission for the Udacity Continuous Control Nanodegree project.

## Setting up environment
In order to run this project you will need to setup a conda environment and to download the UnityML environment.
```
conda create -n deep-reinforcement-learning python=3.6
conda activate deep-reinforcement-learning
conda env update -f environment.yml
```

Env downloads for mac. Do one of the following depending on which :
 - Download [this file](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip) and unzip into `Reacher.app`
 - Download [this file](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip) and unzip into `ReacherMultiple.app`

## Running the project
```
conda activate deep-reinforcement-learning
python -m ppo_continuous_control.train_agent -n 750 --use-multiple-agents
```

You use `python -m ppo_continuous_control.train_agent --help` to find out more about the CLI interface including how to specify
which UnityML environment to use to train your agent.

## Seeing results
Once you have trained agent weights (or if you want to use the checked in weights) you can see how the agent performs as follows:
```
python -m ppo_continuous_control.play_game --use-multiple-agents
```
```
python -m ppo_continuous_control.play_game --use-multiple-agents --actor-weights <filepath> --critic-weights <filepath>
```

## Running tests
This repository contains some unit tests. You can run them from the repo root with
 - `conda activate deep-reinforcement-learning`
 - `pytest .`
 
 
## Environment details
In this environment, a double-jointed arm can move to target locations. 
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
Each action is a vector with four numbers, corresponding to torque applicable to two joints. 
Every entry in the action vector should be a number between -1 and 1.

The multiple agent environment trains the problem using 20 agents within the same unity environment. 
The score is the average score of all 20 agents.

The challenge is considered solved if the average score over the last 100 episodes is greater than 30.
