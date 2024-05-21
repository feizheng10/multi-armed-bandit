from __future__ import division

import time
import numpy as np


class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError


class BernoulliBandit(Bandit):

    def __init__(self, n, probas=None):
        assert probas is None or len(probas) == n
        self.n = n
        if probas is None:
            np.random.seed(int(time.time()))
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        self.best_proba = max(self.probas)

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probas[i]:
            return 1
        else:
            return 0

class NormalBandit(Bandit):

    def __init__(self, n, means=None, stds=None):
        assert means is None or len(means) == n
        assert stds is None or len(stds) == n
        self.n = n
        if means is None:
            np.random.seed(int(time.time()))
            self.means = [np.random.random() for _ in range(self.n)]
        else:
            self.means = means

        if stds is None:
            self.stds = [1.0 for _ in range(self.n)]
        else:
            self.stds = stds

        self.best_mean = max(self.means)

    def generate_reward(self, i):
        return np.random.normal(self.means[i], self.stds[i])