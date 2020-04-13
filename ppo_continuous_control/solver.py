import abc


class Solver(abc.ABC):
    @abc.abstractmethod
    def record_rewards(self, rewards):
        pass

    @abc.abstractmethod
    def is_solved(self) -> bool:
        pass


class NeverSolved(Solver):
    def __init__(self):
        self.latest_rewards = None

    def record_rewards(self, rewards):
        self.latest_rewards = rewards

    def is_solved(self):
        if self.latest_rewards is not None:
            return (self.latest_rewards.sum() / 20) > 30
        return False
