import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from analysis.fitting import nll, run_optimization
from environment.igt_env import get_lhs_samples


def check_sensitivity(mle_params, actions, outcomes, config, horizon):
    """Print NLL change when each parameter is increased by 5%."""
    base = nll(mle_params, actions, outcomes, config, horizon)
    print(f"\nSENSITIVITY CHECK (+5%):")
    for i, name in enumerate(config.param_names):
        p_up = list(mle_params)
        p_up[i] *= 1.05
        delta = nll(p_up, actions, outcomes, config, horizon) - base
        print(f"  {name}: dNLL = {delta:.4f}")


def run_repeated_diagnostics(true_params, n_repeats, config, horizon=0):
    """Run n_repeats independent fits for a single true parameter set.

    Returns:
        Array of shape (n_successful, n_params) with recovered parameters.
    """
    print(f"--- Repeated diagnostics for: {true_params} ---")

    raw_results = Parallel(n_jobs=-1)(
        delayed(run_optimization)(
            true_params=true_params,
            config=config,
            horizon=horizon,
            seed=None,
        )
        for _ in range(n_repeats)
    )

    return np.array([r[1] for r in raw_results if r is not None and r[1] is not None])


def run_reliability_scan(config, n_points, n_repeats, horizon, seed):
    """Sample n_points from the parameter space and test recovery at each point.

    For every sampled parameter vector, runs n_repeats independent fits and
    computes per-point reliability metrics:
      - rmsse_median / rmsse_p90    — overall recovery quality
      - convergence_rate            — fraction of fits that succeeded
      - bias_{name}                 — systematic over/under-estimation
      - std_{name}                  — estimation variance
      - shapiro_p_{name}            — Shapiro-Wilk p-value (>0.05 → normal errors)
      - pct_normal                  — % of parameters with normal error distribution
      - corr_{name_i}_{name_j}      — Pearson r of estimation errors between each
                                       parameter pair (measure of local confounding)
      - max_error_corr              — max |r| across all pairs (scalar confounding index)
      - nll_median                  — median NLL at recovered parameters (fit quality;
                                       high NLL despite low RMSSE → flat landscape)

    Total fits: n_points × n_repeats (all parallelised in one batch).

    Returns:
        DataFrame with one row per point.
    """
    from analysis.fitting import nll as compute_nll
    from itertools import combinations

    points = get_lhs_samples(n_points, config.bounds, seed=seed)
    ranges = np.array([b[1] - b[0] for b in config.bounds])
    param_names = config.param_names
    n_params = len(param_names)

    # Flatten into one task list so joblib has full scheduling control
    tasks = [
        (point_idx, points[point_idx], seed + point_idx * 1000 + rep)
        for point_idx in range(n_points)
        for rep in range(n_repeats)
    ]

    print(f"Reliability scan: {n_points} points × {n_repeats} repeats = {len(tasks)} fits")
    raw_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_optimization)(t[1], config, horizon, seed=t[2])
        for t in tasks
    )

    records = []
    for point_idx in range(n_points):
        true_p = np.array(points[point_idx])
        point_results = raw_results[point_idx * n_repeats : (point_idx + 1) * n_repeats]

        n_attempted      = len(point_results)
        valid_with_rep   = [(rep, r) for rep, r in enumerate(point_results)
                            if r is not None and r[1] is not None]
        est_arr          = np.array([r[1] for _, r in valid_with_rep])

        if len(est_arr) < 3:
            continue  # too few successful fits to compute statistics

        abs_scaled    = np.abs(est_arr - true_p) / ranges
        raw_errors    = est_arr - true_p
        rmsse_per_rep = np.sqrt(np.mean(abs_scaled ** 2, axis=1))

        record = {f"true_{name}": float(true_p[j])
                  for j, name in enumerate(param_names)}
        record["rmsse_median"]     = float(np.median(rmsse_per_rep))
        record["rmsse_p90"]        = float(np.percentile(rmsse_per_rep, 90))
        record["n_successful"]     = len(est_arr)
        record["convergence_rate"] = float(len(est_arr) / n_attempted)

        shapiro_ps = []
        for j, name in enumerate(param_names):
            errors = raw_errors[:, j]
            record[f"bias_{name}"] = float(np.mean(errors))
            record[f"std_{name}"]  = float(np.std(errors))
            _, p_val = stats.shapiro(errors)
            record[f"shapiro_p_{name}"] = float(p_val)
            shapiro_ps.append(p_val)

        record["pct_normal"]       = float(np.mean([p > 0.05 for p in shapiro_ps]) * 100)
        record["shapiro_p_median"] = float(np.median(shapiro_ps))

        # Inter-parameter error correlations (local confounding at this point)
        abs_corrs = []
        for i, j in combinations(range(n_params), 2):
            name_i, name_j = param_names[i], param_names[j]
            r = float(np.corrcoef(raw_errors[:, i], raw_errors[:, j])[0, 1])
            record[f"corr_{name_i}_{name_j}"] = r
            abs_corrs.append(abs(r))
        record["max_error_corr"] = float(max(abs_corrs)) if abs_corrs else float("nan")

        # NLL at recovered parameters — distinguishes flat landscape from optimizer failure
        # valid_with_rep: [(rep_idx, (true_params, est_params)), ...]
        nll_vals = [
            compute_nll(est_p, *_get_data_for_point(true_p, config, horizon,
                                                     seed + point_idx * 1000 + rep_idx),
                        config, horizon)
            for rep_idx, (_, est_p) in valid_with_rep
        ]
        record["nll_median"] = float(np.median(nll_vals))

        records.append(record)

    return pd.DataFrame(records)


def _get_data_for_point(true_p, config, horizon, seed):
    """Re-generate the behavioral data for a given point and seed (matches run_optimization)."""
    from analysis.simulation import generate_data
    return generate_data(true_p, config=config, n_trials=100, horizon=horizon, seed=seed)
