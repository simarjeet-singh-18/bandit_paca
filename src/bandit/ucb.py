"""Upper Confidence Bound bandit (Algorithm 1, UCB-PaCA).

Selection rule (Line 6):
        U_i(t) = mu_i + alpha * sqrt( log t / (N_i + 1) )
Update  (Lines 14-15):
        N_i   <- N_i + 1
        mu_i  <- mu_i + (1 / N_i) * (r - mu_i)

Rewards are non-stationary jumps in validation accuracy; as discussed in the
paper we track empirical means and rely on *relative* differences between arms.
"""

import math
from typing import List

import numpy as np


class UCB:
    def __init__(self, num_arms: int, alpha: float = 1.0):
        self.num_arms = num_arms
        self.alpha = alpha
        self.N = np.zeros(num_arms, dtype=np.float64)   # pull counts
        self.mu = np.zeros(num_arms, dtype=np.float64)  # empirical mean reward
        self.t = 0                                       # global epoch counter

    def scores(self) -> np.ndarray:
        t = max(self.t, 1)
        return self.mu + self.alpha * np.sqrt(math.log(t) / (self.N + 1.0))

    def select(self, select_size: int = 1) -> List[int]:
        """Advance the clock and return the ``select_size`` highest-UCB arms."""
        self.t += 1
        scores = self.scores()
        k = min(select_size, self.num_arms)
        # argsort descending; deterministic tie-breaking on index
        return list(np.argsort(-scores, kind="stable")[:k])

    def update(self, arms: List[int], reward: float) -> None:
        for i in arms:
            self.N[i] += 1.0
            self.mu[i] += (reward - self.mu[i]) / self.N[i]

    def state(self) -> dict:
        return {"N": self.N.tolist(), "mu": self.mu.tolist(), "t": self.t}
