import abc
from collections import deque


class Solver(abc.ABC):
    @abc.abstractmethod
    def record_rewards(self, rewards):
        pass

    @abc.abstractmethod
    def is_solved(self) -> bool:
        pass


class AverageScoreSolver(Solver):
    def __init__(self, num_agents, solved_score, solved_score_period):
        self.latest_rewards = deque(maxlen=solved_score_period)
        self.num_agents = num_agents
        self.solved_score = solved_score

    def record_rewards(self, rewards):
        self.latest_rewards.append(sum(rewards))

    def is_solved(self):
        return (sum(self.latest_rewards) / (self.num_agents * len(self.latest_rewards))) > self.solved_score
