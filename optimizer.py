"""
optimizer.py
------------
Top-level `SimpleGryffin` class tying together:

    data_processing.ParameterSpace   (scaling 7 continuous inputs)
    chimera.Chimera                  (5 targets -> 1 scalar merit)
    surrogate.KernelDensitySurrogate (Bayesian kernel-density model)
    acquisition.optimize_acquisition (Adam-based acquisition search)

Usage pattern (matches the recommend/observe loop of real Gryffin):

    opt = SimpleGryffin(parameters, objectives, general)

    observations = []                       # or your historical data
    for round in range(budget):
        proposals = opt.recommend(observations, num_proposals=2)
        for p in proposals:
            result = run_your_experiment(p)      # -> dict of 5 targets
            p.update(result)
            observations.append(p)

`observations` is always the FULL accumulated history; the optimizer
itself is stateless between calls, exactly like the real Gryffin API.
"""

import numpy as np

from data_processing import ParameterSpace, clean_observations, observations_to_arrays
from chimera import Chimera
from surrogate import KernelDensitySurrogate
from acquisition import optimize_acquisition, select_diverse_batch


DEFAULT_GENERAL = {
    "sampling_strategies": 2,   # number of lambda values / proposals per round
    "num_random_samples": 2000,  # global random search pool size
    "num_restarts": 10,          # how many of those get Adam-refined
    "adam_lr": 0.05,
    "adam_steps": 60,
    "obj_transform": "sqrt",
    "min_rel_dist": 0.05,        # diversity floor between proposals in one batch
    "random_seed": None,
    "verbose": True,
}


class SimpleGryffin:
    def __init__(self, parameters, objectives, general=None):
        """
        Parameters
        ----------
        parameters : list of dict
            [{'name': 'x1', 'low': 0.0, 'high': 1.0}, ... ]  (7 entries)
        objectives : list of dict, IN PRIORITY ORDER (index 0 = most
            important for Chimera's hierarchy)
            [{'name': 'y1', 'goal': 'max', 'tolerance': 0.2, 'absolute': False},
             ...]                                             (5 entries)
        general : dict, optional
            Overrides for DEFAULT_GENERAL.
        """
        self.space = ParameterSpace(parameters)
        self.param_names = self.space.names

        self.obj_names = [o["name"] for o in objectives]
        goals = [o["goal"] for o in objectives]
        tolerances = [o.get("tolerance", 0.2) for o in objectives]
        absolutes = [o.get("absolute", False) for o in objectives]
        self.chimera = Chimera(goals, tolerances, absolutes)

        self.general = dict(DEFAULT_GENERAL)
        if general:
            self.general.update(general)

        self._rng = np.random.default_rng(self.general["random_seed"])

        self.last_merit_ = None     # scalarized merit of the last recommend() call's training data
        self.last_surrogate_ = None

    # ------------------------------------------------------------------
    def _bootstrap(self, num_proposals):
        """Pure random sampling, used when there is no data yet."""
        d = self.space.dim
        U = self._rng.uniform(size=(num_proposals, d))
        X = self.space.from_unit(U)
        return [self.space.vector_to_dict(x) for x in X]

    # ------------------------------------------------------------------
    def recommend(self, observations, num_proposals=None, clean=True):
        """
        Parameters
        ----------
        observations : list of dict
            Full accumulated history. Each dict must contain every
            parameter name and every objective name as numeric keys.
            Pass an empty list on the very first call (cold start).
        num_proposals : int, optional
            How many new parameter sets to propose this round. Defaults
            to general['sampling_strategies'].
        clean : bool
            Run `data_processing.clean_observations` first (drops rows
            with missing/non-numeric/NaN values and exact duplicate
            parameter vectors).

        Returns
        -------
        proposals : list of dict
            Each dict has the 7 parameter names as keys, in REAL units
            (already un-scaled from the internal unit hypercube).
        """
        n_strat = num_proposals or self.general["sampling_strategies"]

        if not observations:
            if self.general["verbose"]:
                print(f"[recommend] no observations yet -> {n_strat} random bootstrap point(s)")
            return self._bootstrap(n_strat)

        if clean:
            observations = clean_observations(
                observations, self.param_names, self.obj_names,
                verbose=self.general["verbose"],
            )
            if not observations:
                if self.general["verbose"]:
                    print("[recommend] nothing left after cleaning -> bootstrap")
                return self._bootstrap(n_strat)

        X_raw, Y_raw = observations_to_arrays(observations, self.param_names, self.obj_names)
        U = self.space.to_unit(X_raw)

        merit = self.chimera.scalarize_and_transform(Y_raw, transform=self.general["obj_transform"])
        self.last_merit_ = merit

        surrogate = KernelDensitySurrogate().fit(U, merit)
        self.last_surrogate_ = surrogate

        lambdas = np.linspace(1.0, -1.0, n_strat)  # +1 exploit ... -1 explore
        cand_x, cand_val = [], []
        for lam in lambdas:
            x, val = optimize_acquisition(
                surrogate, lam, self.space.dim,
                num_random_samples=self.general["num_random_samples"],
                num_restarts=self.general["num_restarts"],
                adam_lr=self.general["adam_lr"],
                adam_steps=self.general["adam_steps"],
                rng=self._rng,
            )
            cand_x.append(x)
            cand_val.append(val)

        kept_idx = select_diverse_batch(cand_x, cand_val, min_rel_dist=self.general["min_rel_dist"])
        U_new = np.array([cand_x[i] for i in kept_idx])
        X_new = self.space.from_unit(U_new)

        if self.general["verbose"]:
            best_seen = observations[int(np.argmin(merit))]
            print(f"[recommend] {len(observations)} obs -> proposing {len(X_new)} point(s); "
                  f"best merit so far = {merit.min():.4f} at { {k: round(best_seen[k], 4) for k in self.obj_names} }")

        return [self.space.vector_to_dict(x) for x in X_new]

    # ------------------------------------------------------------------
    def predict_merit(self, param_dicts):
        """
        Convenience helper: evaluate the current surrogate's predicted
        merit (not the acquisition function) for a list of parameter
        dicts, using whatever surrogate was built on the last
        `recommend()` call. Useful for `gryffin.get_regression_surrogate`
        style inspection.
        """
        if self.last_surrogate_ is None:
            raise RuntimeError("call recommend() at least once before predict_merit()")
        X = np.array([self.space.dict_to_vector(d) for d in param_dicts])
        U = self.space.to_unit(X)
        return self.last_surrogate_.predict(U)
