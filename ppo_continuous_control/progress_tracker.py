import abc
from matplotlib import pyplot as plt
from progressbar import ProgressBar


class ProgressTracker:
    @abc.abstractmethod
    def record_score(self, rewards):
        pass


class ScoreGraphPlotter(ProgressTracker):
    def __init__(self, score_min=None, score_max=None):
        super(ScoreGraphPlotter, self).__init__()

        plt.ion()
        self.scores = []
        self.score_min = score_min
        self.score_max = score_max
        self.times = []
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Episode number")
        self.ax.set_ylabel("Average Agent score")
        (self.line,) = self.ax.plot(self.times, self.scores)

    def __draw(self):
        if len(self.times) > 1:
            self.ax.set_xlim(min(self.times), max(self.times))

        lower_score_bound = self.score_min or min(self.scores)
        upper_score_bound = max(self.score_max or 1, max(self.scores))
        if lower_score_bound != upper_score_bound:
            self.ax.set_ylim(lower_score_bound, upper_score_bound)

        self.line.set_xdata(self.times)
        self.line.set_ydata(self.scores)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def record_score(self, score):
        self.scores.append(score)
        self.times = list(range(len(self.scores)))
        self.__draw()


class ProgressBarTracker(ProgressTracker):
    def __init__(self, n_rollouts):
        super(ProgressTracker, self).__init__()
        self.progress_bar = ProgressBar(max_value=n_rollouts)
        self.index = 0

    def record_score(self, rewards):
        self.progress_bar.update(self.index)
        self.index += 1
