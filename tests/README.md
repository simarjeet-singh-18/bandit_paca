# Tests

Lightweight correctness checks for the algorithmic core.

```bash
# Bandits (UCB + Thompson), random arms, and gradient chains.
# Requires torch installed (src.sensitivity imports it).
python tests/test_core.py

# PaCA forward / merge / re-selection algebra (pure numpy, no torch needed).
python tests/test_paca_math.py
```

Both scripts print `... PASSED` / `... VERIFIED` on success and assert otherwise.
