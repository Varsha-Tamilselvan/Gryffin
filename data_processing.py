"""
data_processing.py
-------------------
Cleaning of raw experimental records and scaling of the 7 continuous
input parameters to the unit hypercube [0, 1]^d, which is the domain
the surrogate and acquisition function operate on internally.

No external dependencies beyond numpy.
"""

import numpy as np


class ParameterSpace:
    """
    Holds the physical bounds of every continuous input parameter and
    converts back and forth between real (physical) units and the
    internal unit-hypercube representation used by the optimizer.

    Parameters
    ----------
    param_specs : list of dict
        Each dict must have keys: 'name', 'low', 'high'.
        Example:
            [{'name': 'temperature', 'low': 20.0, 'high': 120.0}, ...]
    """

    def __init__(self, param_specs):
        if len(param_specs) == 0:
            raise ValueError("param_specs must contain at least one parameter")

        self.names = [p["name"] for p in param_specs]
        self.low = np.array([float(p["low"]) for p in param_specs])
        self.high = np.array([float(p["high"]) for p in param_specs])

        if np.any(self.high <= self.low):
            raise ValueError("Every parameter needs high > low")

        self.dim = len(self.names)

    # ---- unit-hypercube <-> real units -----------------------------
    def to_unit(self, X_real):
        X_real = np.atleast_2d(X_real)
        return (X_real - self.low) / (self.high - self.low)

    def from_unit(self, U):
        U = np.atleast_2d(U)
        return self.low + U * (self.high - self.low)

    # ---- dict <-> vector helpers ------------------------------------
    def dict_to_vector(self, d):
        try:
            return np.array([float(d[name]) for name in self.names])
        except KeyError as e:
            raise KeyError(
                f"Observation is missing required parameter {e}. "
                f"Expected keys: {self.names}"
            )

    def vector_to_dict(self, v):
        return {name: float(val) for name, val in zip(self.names, v)}


def clean_observations(observations, param_names, objective_names,
                        drop_duplicates=True, verbose=True):
    """
    Validate and clean a list of observation dicts before they are fed
    to the optimizer.

    Parameters
    ----------
    observations : list of dict
        Each dict must contain every name in param_names and every name
        in objective_names, mapped to a numeric value.
    param_names : list of str
    objective_names : list of str
    drop_duplicates : bool
        Remove rows whose parameter vector is an exact duplicate of an
        earlier row (keeps the earliest occurrence).
    verbose : bool
        Print a short summary of what was dropped/kept.

    Returns
    -------
    cleaned : list of dict
        The subset of `observations` that passed validation.
    """
    all_keys = param_names + objective_names
    cleaned = []
    seen_param_vectors = []
    n_dropped_missing = 0
    n_dropped_nonnumeric = 0
    n_dropped_nan = 0
    n_dropped_dup = 0

    for row in observations:
        # 1) all required keys present
        if not all(k in row for k in all_keys):
            n_dropped_missing += 1
            continue

        # 2) all values numeric and finite
        try:
            values = [float(row[k]) for k in all_keys]
        except (TypeError, ValueError):
            n_dropped_nonnumeric += 1
            continue

        if any(not np.isfinite(v) for v in values):
            n_dropped_nan += 1
            continue

        # 3) optional de-duplication on the parameter vector only
        if drop_duplicates:
            pvec = tuple(round(float(row[k]), 12) for k in param_names)
            if pvec in seen_param_vectors:
                n_dropped_dup += 1
                continue
            seen_param_vectors.append(pvec)

        cleaned.append(row)

    if verbose:
        print(
            f"[clean_observations] kept {len(cleaned)}/{len(observations)} rows "
            f"(missing_keys={n_dropped_missing}, non_numeric={n_dropped_nonnumeric}, "
            f"nan_or_inf={n_dropped_nan}, duplicates={n_dropped_dup})"
        )

    return cleaned


def observations_to_arrays(observations, param_names, objective_names):
    """
    Convert a list of cleaned observation dicts into two numpy arrays.

    Returns
    -------
    X_raw : ndarray, shape (n, len(param_names))
    Y_raw : ndarray, shape (n, len(objective_names))
    """
    X_raw = np.array([[float(o[p]) for p in param_names] for o in observations])
    Y_raw = np.array([[float(o[t]) for t in objective_names] for o in observations])
    return X_raw, Y_raw
