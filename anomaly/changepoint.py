import bayesian_changepoint_detection.online_changepoint_detection as oncd
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

class ChangePointTS:
    def __init__(self,
                 alpha=2.75,
                 beta=0.13,
                 kappa=1,
                 mu=-1.22,
                 hazard=300,
                 num=10):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.mu = mu
        self.hazard = hazard
        self.num = num

    def detect(self, time_series, idx='', plot=False):
        ts_values = time_series.values
        ts_index = time_series.index
        if not idx:
            idx = ts_index[0]

        R, maxes = oncd.online_changepoint_detection(
            ts_values, partial(oncd.constant_hazard, self.hazard),
            oncd.StudentT(self.alpha, self.beta, self.kappa, self.mu))
        detected = np.where(
            np.concatenate(
                [np.zeros(self.num + 1), R[self.num,
                                           self.num + 1:-1]]) > 0.8)[0]
        detected = ts_index[detected]
        self.breakdowns = detected[detected > idx]

        if plot:
            self._plot(time_series, idx)
        return self.breakdowns

    def _plot(self, time_series, idx):
        fig = plt.figure(figsize=(16, 5))
        plt.plot(time_series)
        plt.axvline(idx, linestyle='--', label='Точка отсечения')
        for i in self.breakdowns:
            plt.axvline(i, c='orange')
        plt.legend()
        plt.show()