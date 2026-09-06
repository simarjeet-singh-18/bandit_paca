"""Gaussian Thompson Sampling bandit (Algorithm 2, TS-PaCA).

Each arm's expected reward is modelled as N(mu_i, sigma_i^2). Per epoch:
    1. sample theta_i ~ N(mu_i, sigma_i^2) for every arm,
    2. select the top-K arms by sampled value,
    3. observe a continuous reward R_t,
    4. exact Bayesian update on the *active* arms with known observation noise:

        tau_post  = 1/sigma_i^2 + 1/sigma_obs^2
        sigma_new^2 = 1/tau_post
        mu_i      = sigma_new^2 * ( mu_i/sigma_i^2 + R_t/sigma_obs^2 )
        sigma_i   = sqrt(sigma_new^2)
"""

import math
from typing import List

import numpy as np


class GaussianThompsonSampling:
    def __init__(self, num_arms: int, mu0: float = 0.0, sigma0: float = 1.0,
                 sigma_obs: float = 1.0, rng: np.random.Generator = None):
        self.num_arms = num_arms
        self.mu = np.full(num_arms, float(mu0), dtype=np.float64)
        self.sigma = np.full(num_arms, float(sigma0), dtype=np.float64)
        self.sigma_obs = float(sigma_obs)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.t = 0

    def select(self, select_size: int = 1) -> List[int]:
        self.t += 1
        theta = self.rng.normal(self.mu, self.sigma)
        k = min(select_size, self.num_arms)
        return list(np.argsort(-theta, kind="stable")[:k])

    def update(self, arms: List[int], reward: float) -> None:
        obs_prec = 1.0 / (self.sigma_obs ** 2)
        for i in arms:
            prior_prec = 1.0 / (self.sigma[i] ** 2)
            tau_post = prior_prec + obs_prec
            sigma_new2 = 1.0 / tau_post
            self.mu[i] = sigma_new2 * (self.mu[i] * prior_prec + reward * obs_prec)
            self.sigma[i] = math.sqrt(sigma_new2)

    def state(self) -> dict:
        return {"mu": self.mu.tolist(), "sigma": self.sigma.tolist(), "t": self.t}
