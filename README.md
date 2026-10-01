# SimpleGryffin — from-scratch, pure NumPy/SciPy implementation

A standalone reimplementation of the Gryffin/Phoenics/Chimera pipeline for
**7 continuous inputs → 5 targets**, built directly from the equations in the
three papers (no cloning of the original Cython/PyTorch/PyMC3 repo, no
compiled extensions — just `numpy`).

```
data_processing.py   cleaning + scaling (7 continuous inputs -> [0,1]^7)
chimera.py            5 targets -> 1 scalar merit  (Hase et al., Chem. Sci. 2018)
surrogate.py           Bayesian kernel-density surrogate (Hase et al., ACS Cent. Sci. 2018)
acquisition.py          Adam-optimized acquisition function + batch diversity
optimizer.py            SimpleGryffin: ties it all together, recommend()/update loop
example_run.py          end-to-end demo on a synthetic 7D -> 5D problem
```

## Pipeline, step by step

1. **Data cleaning & scaling** (`data_processing.py`)
   `clean_observations()` drops rows with missing keys, non-numeric values,
   NaN/Inf, or exact duplicate parameter vectors. `ParameterSpace` rescales
   your 7 real-valued inputs to the unit hypercube `[0,1]^7` the optimizer
   works in internally, and back again when it hands you a proposal.

2. **Multi-objective → single-objective (Chimera)** (`chimera.py`)
   Implements Eqs. 1–5 of Häse, Roch & Aspuru-Guzik, *Chem. Sci.* 2018, 9,
   7642: objectives are ranked in a **priority hierarchy** (you decide the
   order — most important first). Each objective gets a tolerance (how much
   it's allowed to degrade so a lower-priority objective can improve),
   either as a fraction of its observed range (`absolute: False`) or as a
   literal threshold in its own units (`absolute: True`, e.g. "yield ≥
   0.9"). The result is one scalar merit, normalized to `[0,1]` and
   `sqrt`-transformed to sharpen the landscape near the optimum — this is
   what the surrogate is actually trained on.

3. **Bayesian surrogate** (`surrogate.py`)
   A kernel-density regression surrogate: one isotropic Gaussian kernel
   centered at each observation, with precision `τ_n = 12·n²` (the
   posterior-mean precision schedule the Phoenics SI, Sec. VI.C, found to
   perform best). This reproduces — deterministically, with plain NumPy —
   the effective behaviour of the original PyMC3/PyTorch Bayesian Neural
   Network for a **purely continuous** parameter space like yours (no
   categorical variables, so there's nothing for the BNN's descriptor
   machinery to do beyond picking kernel locations/precision). The upside:
   it's fully analytic and differentiable, see point 4.

4. **Acquisition function + optimizer** (`acquisition.py`)
   The exact Phoenics/Gryffin acquisition function (Eq. 12 in Häse et al.
   2018 / Eq. 1 in Hickman et al. 2022):
   ```
   α(x) = (Σ_k f_k p_k(x) + λ·p_uniform) / (Σ_k p_k(x) + p_uniform)
   ```
   `λ` sweeps from `+1` (exploit) to `-1` (explore) across however many
   proposals you ask for per round. Optimized in two stages exactly as
   described in the papers' Methods: (1) a cheap vectorized random search
   over the unit cube, (2) local refinement of the best candidates with
   **Adam**, using the exact analytic gradient of α (derived and verified
   against finite differences — see below). A simple greedy distance filter
   keeps proposals in the same batch from collapsing onto each other.

5. **Recommend / update loop** (`optimizer.py`)
   `SimpleGryffin.recommend(observations, num_proposals)`:
   - no `observations` yet → returns random bootstrap point(s)
   - otherwise → cleans data → Chimera-scalarizes → refits the surrogate
     from scratch on the full history → optimizes the acquisition function
     for each λ → returns new parameter dict(s) in your real units.

   You then run your real experiment(s), append the 5 results to each
   returned dict, push it onto your `observations` list, and call
   `recommend()` again with the **full** accumulated history — exactly like
   the real Gryffin API (the optimizer itself is stateless between calls).

## Quick start

```python
from optimizer import SimpleGryffin

parameters = [
    {"name": "x1", "low": 0.0, "high": 10.0},
    # ... 6 more continuous parameters, your real physical bounds
]

objectives = [   # ORDER = priority hierarchy for Chimera, most important first
    {"name": "y1", "goal": "max", "tolerance": 0.2, "absolute": False},
    {"name": "y2", "goal": "min", "tolerance": 0.2, "absolute": False},
    {"name": "y3", "goal": "max", "tolerance": 0.2, "absolute": False},
    {"name": "y4", "goal": "min", "tolerance": 0.2, "absolute": False},
    {"name": "y5", "goal": "max", "tolerance": 0.2, "absolute": False},
]

opt = SimpleGryffin(parameters, objectives, general={"sampling_strategies": 2})

observations = []                       # or load your historical CSV here
for round in range(30):
    proposals = opt.recommend(observations, num_proposals=2)
    for p in proposals:
        result = run_your_real_experiment(p)   # -> {'y1':.., ..., 'y5':..}
        p.update(result)
        observations.append(p)
```

Run `python3 example_run.py` to see this on a synthetic 7D→5D test function —
the top-priority objective climbs steadily toward its optimum while Chimera
keeps the other four in check, across 12 rounds / 24 evaluations.

## Deliberate simplifications vs. the original Cython/PyTorch repo

- **No MCMC/variational BNN.** The real Gryffin retrains a small Bayesian
  neural net (NUTS or variational inference) every call to place kernels
  and sample a precision from `Gamma(12n², 1)`. For a purely continuous
  space this BNN's kernel locations converge to the observations
  themselves; we use that limit directly (kernel per observation, `τ_n =
  12n²` = the Gamma prior's mean) instead of running an actual sampler.
  This is faithful and *removes* the PyMC3/PyTorch dependency entirely.
- **No categorical/descriptor support.** Since your data is 7 continuous
  inputs, the categorical (`na"ive`/`static`/`dynamic` Gryffin) machinery
  for descriptor-guided simplex metrics was left out. If you later add
  categorical variables (e.g. catalyst choice), that's the next piece to
  add (Sec. III of the Gryffin paper) — ask and I'll build it.
- **Adam's analytic gradient** was derived from the chain rule on the
  acquisition function and verified against finite differences
  (`max abs diff ≈ 4e-11` in testing) — see the gradient check note.
- **Known constraints** (Hickman et al. 2022) — arbitrary feasibility
  functions `c(x)` via rejection sampling / projection to the feasibility
  boundary — are not yet implemented. Straightforward to bolt onto the
  random-search stage of `acquisition.optimize_acquisition` if needed.

## Extending

- **More proposals per round / parallel batches**: raise
  `general["sampling_strategies"]`.
- **Different objective priorities**: just reorder the `objectives` list —
  Chimera always treats index 0 as most important.
- **Inspect the surrogate**: `opt.predict_merit(list_of_param_dicts)` gives
  the current kernel-regression prediction (not the acquisition value) for
  arbitrary points, analogous to `gryffin.get_regression_surrogate`.
- **Persisting state between sessions**: this implementation is stateless
  by design — just `pickle.dump(observations, ...)` and reload it on the
  next run; call `recommend()` with the reloaded list.
