"""Correctness checks for bandits, arm construction, and gradient chains."""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bandit.ucb import UCB
from src.bandit.thompson import GaussianThompsonSampling
from src.bandit.arms import build_random_arms, union_arms, arms_to_layer_dims
from src.sensitivity import build_gradient_chains


def test_ucb():
    ucb = UCB(num_arms=4, alpha=1.0)
    for _ in range(40):
        sel = ucb.select(1)
        ucb.update(sel, 1.0 if sel[0] == 2 else 0.0)
    assert ucb.N[2] == max(ucb.N)
    u = UCB(2, alpha=2.0); u.N = np.array([3., 1.]); u.mu = np.array([0.5, 0.2]); u.t = 9
    sc = u.scores()
    assert abs(sc[0] - (0.5 + 2.0 * math.sqrt(math.log(9) / 4))) < 1e-9
    assert abs(sc[1] - (0.2 + 2.0 * math.sqrt(math.log(9) / 2))) < 1e-9
    print("UCB OK")


def test_thompson():
    ts = GaussianThompsonSampling(1, mu0=0.0, sigma0=1.0, sigma_obs=1.0)
    ts.update([0], 2.0)  # posterior: var=0.5, mu=1.0
    assert abs(ts.mu[0] - 1.0) < 1e-9 and abs(ts.sigma[0] - math.sqrt(0.5)) < 1e-9
    print("Thompson OK")


def test_random_arms():
    dims = {"L0": 20, "L1": 12}
    arms = build_random_arms(dims, num_arms=3, rank=4, rng=np.random.default_rng(1))
    for name in dims:
        seen = [c for a in arms for c in a.get(name, [])]
        assert len(seen) == len(set(seen))  # disjoint / non-overlapping
    assert all(len(a["L0"]) == 4 for a in arms)
    _ = union_arms(arms, [0, 2]); _ = arms_to_layer_dims(arms)
    print("random arms OK")


def test_gradient_chains():
    S = {
        "A": np.abs(np.random.default_rng(2).normal(size=(8, 6))),
        "B": np.abs(np.random.default_rng(3).normal(size=(4, 5))),
    }
    chains = build_gradient_chains(S, ["A", "B"], num_chains=3, chain_width=1)
    assert len(chains) == 3
    for c in chains:
        assert len(c["A"]) == 1 and len(c["B"]) == 1
        i_star = int(np.argmax(S["A"][:, c["A"][0]]))
        assert c["B"][0] == i_star % S["B"].shape[1]  # Eq. 6 propagation
    assert len({c["A"][0] for c in chains}) == 3  # distinct starts
    wide = build_gradient_chains(S, ["A", "B"], num_chains=2, chain_width=3)
    assert all(len(c["A"]) == 3 and len(c["B"]) == 3 for c in wide)
    print("gradient chains OK")


if __name__ == "__main__":
    test_ucb(); test_thompson(); test_random_arms(); test_gradient_chains()
    print("\nALL CORE LOGIC TESTS PASSED")
