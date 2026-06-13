import pandas as pd
from joblib import Parallel, delayed
from environment.igt_env import get_lhs_samples
from analysis.simulation import generate_data


def run_psp_analysis(n_samples, config, n_trials, horizon, strategy, seed=42):
    """Parameter Space Partitioning: map sampled parameter vectors to behavioral patterns.

    Args:
        n_samples: number of LHS samples
        config: AgentConfig
        n_trials: game length per simulation
        horizon: look-ahead depth
        strategy: BehaviorStrategy instance
        seed: base random seed

    Returns:
        DataFrame with param columns + 'Pattern' column.
    """
    param_samples = get_lhs_samples(n_samples, config.bounds, seed=seed)

    def single_sim(p, i):
        actions, _ = generate_data(p, config=config, n_trials=n_trials, horizon=horizon, seed=seed + i)
        label = strategy.classify(actions)
        return list(p) + [label]

    results = Parallel(n_jobs=-1)(
        delayed(single_sim)(p, i) for i, p in enumerate(param_samples)
    )

    columns = config.param_names + ["Pattern"]
    return pd.DataFrame(results, columns=columns)


def summarize_psp(df, mode_name):
    """Print pattern distribution table and return (summary_df, percents Series)."""
    counts = df["Pattern"].value_counts()
    percents = df["Pattern"].value_counts(normalize=True) * 100

    summary_table = pd.DataFrame({
        "Agent Count": counts,
        "Share [%]": percents.round(2),
    })

    print(f"\n=== PSP analysis: {mode_name} ===")
    print(summary_table)

    return summary_table, percents
