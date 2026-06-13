import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from analysis.fitting import run_optimization
from environment.igt_env import get_lhs_samples


def run_parameter_recovery(config, n_agents, horizon, n_trials, seed):
    """Generate n_agents synthetic datasets and recover parameters in parallel.

    Returns:
        (df, true_res, rec_res) or None if all fits failed.
        df contains true_{name} and est_{name} columns for each parameter.
    """
    true_params_list = get_lhs_samples(n_agents, config.bounds, seed=seed)

    raw_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_optimization)(
            true_params=tp,
            config=config,
            horizon=horizon,
            seed=seed + i,
        )
        for i, tp in enumerate(tqdm(true_params_list, desc="Parameter Recovery"))
    )

    valid = [r for r in raw_results if r is not None and r[1] is not None]
    if not valid:
        return None

    true_res, rec_res = zip(*valid)
    true_arr = np.array(true_res)
    rec_arr = np.array(rec_res)

    df = pd.DataFrame()
    for i, name in enumerate(config.param_names):
        df[f"true_{name}"] = true_arr[:, i]
        df[f"est_{name}"] = rec_arr[:, i]

    return df, true_res, rec_res


def run_distributional_recovery(config, n_agents, n_repeats, horizon, seed):
    """For each of n_agents, run n_repeats independent fits to study error distributions."""
    print(f"--- Distributional Recovery: {n_agents} agents × {n_repeats} repeats ---")

    base_agents = get_lhs_samples(n_agents, config.bounds, seed=seed)

    tasks = [
        (agent_id, true_p, seed + agent_id * 1000 + rep)
        for agent_id, true_p in enumerate(base_agents)
        for rep in range(n_repeats)
    ]

    raw_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_optimization)(
            true_params=t[1],
            config=config,
            horizon=horizon,
            seed=t[2],
        )
        for t in tasks
    )

    records = []
    for i, res in enumerate(raw_results):
        if res is not None and res[1] is not None:
            agent_id, true_p, _ = tasks[i]
            est_p = res[1]
            record = {"agent_id": agent_id}
            for j, name in enumerate(config.param_names):
                record[f"true_{name}"] = true_p[j]
                record[f"est_{name}"] = est_p[j]
                record[f"err_{name}"] = est_p[j] - true_p[j]
            records.append(record)

    return pd.DataFrame(records)


def run_comparative_recovery(config_a, config_b, n_agents, horizon, seed):
    """Self-recovery for two models on the same shared seeds and environment.

    Both models use identical per-agent seeds and EnvironmentBechara card sequences,
    so the comparison is apples-to-apples.

    Returns:
        (df_a, df_b) — one DataFrame per model with true_{name} / est_{name} columns.
    """
    lhs_a = get_lhs_samples(n_agents, config_a.bounds, seed=seed)
    lhs_b = get_lhs_samples(n_agents, config_b.bounds, seed=seed)

    def _recover(config, lhs):
        raw = Parallel(n_jobs=-1, verbose=0)(
            delayed(run_optimization)(
                true_params=tp,
                config=config,
                horizon=horizon,
                seed=seed + i,
            )
            for i, tp in enumerate(lhs)
        )
        valid = [r for r in raw if r is not None and r[1] is not None]
        if not valid:
            return None
        true_arr = np.array([r[0] for r in valid])
        rec_arr  = np.array([r[1] for r in valid])
        df = pd.DataFrame()
        for i, name in enumerate(config.param_names):
            df[f"true_{name}"] = true_arr[:, i]
            df[f"est_{name}"]  = rec_arr[:, i]
        return df

    df_a = _recover(config_a, lhs_a)
    df_b = _recover(config_b, lhs_b)
    return df_a, df_b


def compute_recovery_errors(df, config):
    """Add scaled error columns and global RMSSE to a recovery DataFrame.

    Modifies df in-place and returns it.
    """
    ranges = {name: b[1] - b[0] for name, b in zip(config.param_names, config.bounds)}

    for name in config.param_names:
        df[f"err_{name}"] = df[f"est_{name}"] - df[f"true_{name}"]
        df[f"abs_err_{name}"] = np.abs(df[f"err_{name}"]) / ranges[name]

    scaled_sq = [df[f"abs_err_{name}"] ** 2 for name in config.param_names]
    df["global_error"] = np.sqrt(np.mean(scaled_sq, axis=0))

    return df
