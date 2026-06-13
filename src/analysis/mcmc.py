import numpy as np
import emcee
from joblib import Parallel, delayed
from analysis.fitting import nll, fit_model


def _kl_posterior_uniform(samples, lo, hi):
    """KL(posterior || Uniform(lo, hi)) estimated from samples via KDE.

    Uses the Monte Carlo estimator:  KL = E_{x~P}[log P(x)] + log(hi - lo)
    where log P(x) is evaluated via Gaussian KDE fitted to the samples.

    Returns nats; clipped at 0 to remove negative estimation artefacts.
    Interpretation: 0 = posterior indistinguishable from prior (no info),
    higher = more information gained from data.
    """
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples, bw_method="scott")
    log_post = kde.logpdf(samples)
    kl = float(np.mean(log_post)) + np.log(hi - lo)
    return max(0.0, kl)


def _log_posterior(params, actions, outcomes, config, horizon):
    """Log-posterior with uniform (flat) priors within parameter bounds."""
    for i, (lo, hi) in enumerate(config.bounds):
        if not (lo <= params[i] <= hi):
            return -np.inf
    return -nll(params, actions, outcomes, config, horizon)


def run_mcmc(actions, outcomes, config, horizon=0,
             n_walkers=32, n_steps=2000, n_burnin=500, seed=None):
    """Sample the posterior using an affine-invariant ensemble sampler (emcee).

    Walkers are initialised around the MLE estimate (if found) with small
    perturbations; falling back to a random spread within bounds.

    Args:
        actions, outcomes : game data from generate_data()
        config            : AgentConfig
        horizon           : look-ahead depth
        n_walkers         : number of parallel MCMC chains (must be ≥ 2 × n_params)
        n_steps           : production steps per walker
        n_burnin          : burn-in steps (discarded before production)
        seed              : RNG seed

    Returns:
        sampler      : emcee.EnsembleSampler (full chain history)
        flat_samples : ndarray of shape (n_walkers × n_steps, n_params)
        chain        : ndarray of shape (n_steps, n_walkers, n_params)
    """
    n_params = len(config.bounds)
    rng = np.random.default_rng(seed)

    # --- Walker initialisation ---
    # Try to start near the MLE; if fitting fails, spread across the prior.
    mle = fit_model(actions, outcomes, config, horizon=horizon, method="MLE", seed=seed)
    if mle is not None:
        center = np.array(mle)
        spread = np.array([(b[1] - b[0]) * 0.05 for b in config.bounds])
        print("Walkers initialised near MLE.")
    else:
        center = np.array([(b[0] + b[1]) / 2 for b in config.bounds])
        spread = np.array([(b[1] - b[0]) * 0.3 for b in config.bounds])
        print("MLE failed — walkers spread across prior.")

    p0 = center + rng.standard_normal((n_walkers, n_params)) * spread
    for i, (lo, hi) in enumerate(config.bounds):
        p0[:, i] = np.clip(p0[:, i], lo + 1e-6, hi - 1e-6)

    sampler = emcee.EnsembleSampler(
        n_walkers, n_params, _log_posterior,
        args=(actions, outcomes, config, horizon),
    )

    # Burn-in
    print(f"Burn-in: {n_burnin} steps × {n_walkers} walkers …")
    state = sampler.run_mcmc(p0, n_burnin, progress=True)
    sampler.reset()

    # Production
    print(f"Sampling: {n_steps} steps × {n_walkers} walkers …")
    sampler.run_mcmc(state, n_steps, progress=True)

    flat_samples = sampler.get_chain(flat=True)   # (n_walkers × n_steps, n_params)
    chain        = sampler.get_chain()             # (n_steps, n_walkers, n_params)

    return sampler, flat_samples, chain


def compute_mcmc_diagnostics(sampler, flat_samples, config):
    """Compute convergence and identifiability diagnostics.

    Returns a dict keyed by parameter name, each with:
      ess           effective sample size (via autocorrelation)
      mean          posterior mean
      std           posterior standard deviation
      ci_95         95% credible interval (lo, hi)
      shrinkage     1 − posterior_CI_width / prior_width  ∈ [0, 1]
                    ≈ 0 → not identified;  ≈ 1 → well identified
      kl_div        KL(posterior || prior) in nats (KDE estimator)
                    0 = posterior = prior (no info); higher = better identified
      max_post_corr max |r| with any other parameter (confounding index)
    """
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        ess = flat_samples.shape[0] / (2 * tau)
    except Exception:
        ess = np.full(len(config.param_names), np.nan)

    prior_widths = np.array([b[1] - b[0] for b in config.bounds])
    ci_lo = np.percentile(flat_samples, 2.5,  axis=0)
    ci_hi = np.percentile(flat_samples, 97.5, axis=0)
    ci_widths = ci_hi - ci_lo

    # Posterior correlations (prior-free measure of parameter confounding)
    post_corr = np.corrcoef(flat_samples.T)

    diags = {}
    for i, name in enumerate(config.param_names):
        lo, hi = config.bounds[i]
        diags[name] = {
            "ess":            float(ess[i]) if not np.isnan(ess[i]) else None,
            "mean":           float(flat_samples[:, i].mean()),
            "std":            float(flat_samples[:, i].std()),
            "ci_95":          (float(ci_lo[i]), float(ci_hi[i])),
            # Prior-dependent — sensitive to bounds width; use as secondary check only
            "shrinkage":      float(np.clip(1.0 - ci_widths[i] / prior_widths[i], 0, 1)),
            # Information-theoretic identifiability: KL(posterior || prior) in nats
            "kl_div":         _kl_posterior_uniform(flat_samples[:, i], lo, hi),
            # Max |r| with any other parameter — detects posterior degeneracy
            "max_post_corr":  float(np.max(np.abs(np.delete(post_corr[i], i)))),
        }

    return diags


def _mcmc_scan_one_point(true_p, config, horizon, seed, n_walkers, n_steps, n_burnin):
    """Run MCMC for one LHS point and return a flat record dict."""
    from itertools import combinations
    from analysis.simulation import generate_data

    actions, outcomes = generate_data(true_p, config=config,
                                      n_trials=150, horizon=horizon, seed=seed)
    try:
        sampler, flat_samples, _ = run_mcmc(
            actions, outcomes, config,
            horizon=horizon, n_walkers=n_walkers,
            n_steps=n_steps, n_burnin=n_burnin, seed=seed,
        )
        diags = compute_mcmc_diagnostics(sampler, flat_samples, config)
    except Exception:
        return None

    param_names = config.param_names
    post_corr = np.corrcoef(flat_samples.T)

    record = {f"true_{name}": float(true_p[j])
              for j, name in enumerate(param_names)}

    for name in param_names:
        d = diags[name]
        record[f"kl_div_{name}"]   = d["kl_div"]
        record[f"post_std_{name}"] = d["std"]
        record[f"ess_{name}"]      = d["ess"] if d["ess"] is not None else float("nan")

    abs_corrs = []
    for i, j in combinations(range(len(param_names)), 2):
        r = float(post_corr[i, j])
        record[f"post_corr_{param_names[i]}_{param_names[j]}"] = r
        abs_corrs.append(abs(r))

    record["max_post_corr"]  = float(max(abs_corrs)) if abs_corrs else float("nan")
    record["mean_kl_div"]   = float(np.mean([diags[n]["kl_div"] for n in param_names]))
    record["min_ess"]       = float(np.nanmin([diags[n]["ess"] or float("nan")
                                                   for n in param_names]))
    return record


def run_mcmc_reliability_scan(config, n_points, horizon, seed,
                               n_walkers=32, n_steps=1000, n_burnin=300,
                               n_jobs=-1):
    """MCMC-based reliability scan across the parameter space.

    For each of n_points LHS-sampled parameter vectors, runs a full MCMC
    and extracts posterior-based identifiability metrics. Methodologically
    stronger than the MLE-based reliability scan: measures how much the data
    actually constrains each parameter, not just whether the optimizer converges.

    Key output columns:
      kl_div_{name}                   — KL(posterior||prior) per parameter in nats (0=no info)
      post_std_{name}                 — posterior SD (prior-free spread)
      ess_{name}                      — effective sample size
      post_corr_{name_i}_{name_j}    — posterior correlation (confounding)
      max_post_corr                   — max |r| across all pairs (scalar confounding index)
      mean_kl_div                     — average KL divergence across all parameters
      min_ess                         — worst ESS across parameters

    Args:
        n_walkers: keep ≥ 2 × n_params; 4 × n_params is a safe default
        n_steps:   1000 is usually sufficient for a scan; use 2000 for final runs
        n_burnin:  300 for scans; 500 for final runs
        n_jobs:    joblib parallelism across points (-1 = all CPUs)
                   Note: emcee itself is single-threaded, so this is safe.

    Returns:
        pd.DataFrame with one row per point.
    """
    import pandas as pd
    from environment.igt_env import get_lhs_samples

    points = get_lhs_samples(n_points, config.bounds, seed=seed)
    print(f"MCMC reliability scan: {n_points} points  "
          f"({n_walkers} walkers × {n_steps} steps × {n_burnin} burn-in)")

    records = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_mcmc_scan_one_point)(
            true_p=np.array(points[i]),
            config=config, horizon=horizon,
            seed=seed + i,
            n_walkers=n_walkers, n_steps=n_steps, n_burnin=n_burnin,
        )
        for i in range(n_points)
    )

    valid = [r for r in records if r is not None]
    print(f"Successful: {len(valid)} / {n_points}")
    return pd.DataFrame(valid)


def print_mcmc_diagnostics(diags):
    """Pretty-print the diagnostics table."""
    header = (f"{'Parameter':<25} {'Mean':>8} {'Std':>8} {'CI 95%':>20}"
              f" {'KL (nats)':>10} {'MaxCorr':>8} {'ESS':>8}")
    print("\n" + "=" * len(header))
    print("MCMC POSTERIOR DIAGNOSTICS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, d in diags.items():
        ci_str  = f"[{d['ci_95'][0]:.3f}, {d['ci_95'][1]:.3f}]"
        ess_str = f"{d['ess']:.0f}" if d["ess"] is not None else "n/a"
        flags = ""
        if d["kl_div"] < 1.0:
            flags += " ⚠ low KL"
        if d["max_post_corr"] > 0.7:
            flags += " ⚠ confounded"
        print(f"  {name:<23} {d['mean']:>8.3f} {d['std']:>8.3f} {ci_str:>20}"
              f" {d['kl_div']:>10.3f} {d['max_post_corr']:>8.3f} {ess_str:>8}{flags}")
    print("=" * len(header))
    print("KL < 1.0 nat ⚠ → data provides little information beyond the prior.")
    print("MaxCorr > 0.7 ⚠ → parameter is confounded with another.\n")
