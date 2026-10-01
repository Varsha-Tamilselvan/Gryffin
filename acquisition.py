"""
acquisition.py
--------------
Optimizes the Phoenics/Gryffin acquisition function

    alpha(x) = (sum_k f_k p_k(x) + lam * p_uniform) /
               (sum_k p_k(x) + p_uniform)

(to be MINIMIZED, per Eq. 1 of Hickman et al. 2022 and Eq. 12 of
Hase et al. 2018) using the same two-stage strategy described in those
papers' Methods sections:

    1. Global search: draw many points uniformly at random over the
       unit hypercube and evaluate alpha on all of them cheaply
       (vectorized).
    2. Local refinement: take the best few candidates and refine them
       with a first-order gradient method. The Gryffin papers use Adam
       by default (Kingma & Ba, 2017) for continuous parameters, so we
       implement Adam from scratch here using the exact analytic
       gradient supplied by `surrogate.value_and_grad_acquisition`.

A simple distance-based diversity penalty (Sec. "sample selector" of
the Gryffin source, and the self-avoidance behaviour discussed in the
known-constraints paper, Fig. S3) is applied when multiple proposals
are requested, so that a batch of recommendations does not collapse
onto nearly the same point.
"""

import numpy as np


def adam_minimize(value_and_grad_fn, x0, lr=0.05, steps=60,
                   beta1=0.9, beta2=0.999, eps=1e-8, bounds=(0.0, 1.0)):
    """
    Minimal from-scratch Adam optimizer (Kingma & Ba, 2017) operating
    on a single point x in R^d, with simple clipping back into
    `bounds` after every step (the parameter space here is always the
    unit hypercube).

    `value_and_grad_fn(x) -> (value, grad)` must return both the
    scalar objective value and its gradient at x.

    Returns
    -------
    best_x : ndarray
    best_val : float
        The best (lowest) value and position encountered along the
        trajectory (not necessarily the final iterate), matching the
        "keep the best observed sample" behaviour used for acquisition
        optimization in the Gryffin/Phoenics source.
    """
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    lo, hi = bounds

    best_x = x.copy()
    best_val, _ = value_and_grad_fn(x)

    for t in range(1, steps + 1):
        val, grad = value_and_grad_fn(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        x = np.clip(x, lo, hi)

    # check the final point too
    final_val, _ = value_and_grad_fn(x)
    if final_val < best_val:
        best_val = final_val
        best_x = x.copy()

    return best_x, best_val


def optimize_acquisition(surrogate, lam, dim, num_random_samples=2000,
                          num_restarts=10, adam_lr=0.05, adam_steps=60,
                          rng=None):
    """
    Full two-stage acquisition optimization for one sampling strategy
    (one value of lambda): random global search -> Adam local refine.

    Returns
    -------
    best_x : ndarray, shape (d,)   in the unit hypercube
    best_val : float               acquisition value at best_x
    """
    rng = rng or np.random.default_rng()

    # Stage 1: global random search, vectorized
    Xr = rng.uniform(size=(num_random_samples, dim))
    alphas = surrogate.alpha_batch(Xr, lam)
    order = np.argsort(alphas)
    top_idx = order[: min(num_restarts, num_random_samples)]

    # Stage 2: local Adam refinement from each of the top candidates
    best_x, best_val = None, np.inf
    for i in top_idx:
        x0 = Xr[i]
        xr, val = adam_minimize(
            lambda x: surrogate.value_and_grad_acquisition(x, lam),
            x0, lr=adam_lr, steps=adam_steps, bounds=(0.0, 1.0),
        )
        if val < best_val:
            best_val = val
            best_x = xr

    return best_x, best_val


def select_diverse_batch(candidates, candidate_vals, min_rel_dist=0.05):
    """
    Greedy diversity filter for a batch of proposed points (one per
    sampling strategy). If two candidates are closer than
    `min_rel_dist` (as a fraction of the unit-hypercube diagonal),
    the one with the worse (higher) acquisition value is nudged away
    by re-drawing it is NOT done here to keep things simple and
    deterministic; instead we just flag the collision so the caller
    can decide what to do (e.g. increase sampling_strategies).

    This mirrors the self-avoidance behaviour Gryffin exhibits natively
    through its sample selector (see known-constraints paper, Fig. S3)
    -- the kernel density already pushes new points away from existing
    observations, so this helper's job is only to avoid two *proposals
    in the same batch* landing on top of each other.

    Returns
    -------
    kept_idx : list of int
        Indices (into `candidates`) that are mutually far enough apart.
    """
    candidates = np.asarray(candidates, dtype=float)
    n = len(candidates)
    if n <= 1:
        return list(range(n))

    order = np.argsort(candidate_vals)  # best (lowest alpha) first
    kept = []
    for idx in order:
        x = candidates[idx]
        too_close = False
        for kidx in kept:
            if np.linalg.norm(x - candidates[kidx]) < min_rel_dist:
                too_close = True
                break
        if not too_close:
            kept.append(idx)

    if not kept:
        kept = [order[0]]
    return kept
