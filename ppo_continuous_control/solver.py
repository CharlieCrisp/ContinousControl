import abc


class Solver(abc.ABC):
    @abc.abstractmethod
    def record_rewards(self, rewards):
        pass

    @abc.abstractmethod
    def is_solved(self) -> bool:
        pass


class NeverSolved(Solver):
    def record_rewards(self, rewards):
        pass

    def is_solved(self):
        return False
