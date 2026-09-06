import numpy as np
# Mirror PaCALinear.forward / set_active / merge exactly in numpy to validate algebra.
rng = np.random.default_rng(0)
out_f, in_f = 5, 8
W0 = rng.normal(size=(out_f, in_f))         # frozen full weight (buffer)
x = rng.normal(size=(4, in_f))              # batch of 4

def paca_forward(W, delta, idx, x):
    out = x @ W.T                            # F.linear(x, W)
    if delta is not None:
        x_sel = x[:, idx]                    # x.index_select(-1, idx)
        frozen_sel = W[:, idx]               # W.index_select(1, idx)
        out = out + x_sel @ (delta - frozen_sel).T
    return out

# 1) at init delta == frozen columns -> forward equals full frozen forward
idx = np.array([1, 3, 6])
delta = W0[:, idx].copy()
assert np.allclose(paca_forward(W0, delta, idx, x), x @ W0.T), "init correction not zero"
print("init correction == 0 : OK")

# 2) after training delta -> forward equals effective weight (cols replaced)
delta_trained = delta + rng.normal(size=delta.shape)      # pretend gradient step
W_eff = W0.copy(); W_eff[:, idx] = delta_trained
assert np.allclose(paca_forward(W0, delta_trained, idx, x), x @ W_eff.T), "forward != effective"
print("forward == effective-weight forward : OK")

# 3) merge writes trained cols back; frozen columns untouched
W = W0.copy()
W[:, idx] = delta_trained                     # merge()
assert np.allclose(W[:, idx], delta_trained)
mask = np.ones(in_f, bool); mask[idx] = False
assert np.allclose(W[:, mask], W0[:, mask]), "frozen columns changed!"
print("merge persists trained cols, frozen cols intact : OK")

# 4) reselect a NEW set starts from current (merged) values -> continuity
idx2 = np.array([0, 3, 7])                     # overlaps col 3 (now trained)
delta2 = W[:, idx2].copy()                     # set_active extracts current values
assert np.allclose(paca_forward(W, delta2, idx2, x), x @ W.T), "reselect init not continuous"
# col 3 retains its trained value in the new selection
assert np.allclose(delta2[:, list(idx2).index(3)], delta_trained[:, list(idx).index(3)])
print("reselect continuity (learned col 3 preserved) : OK")

print("\nPaCA MATH IDENTITY VERIFIED")
