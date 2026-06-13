import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot is imported

import numpy as np
import pandas as pd

from config import (AGENT_CONFIGS, ACTIVE_AGENT, MODE, REPLOT,
                    N_AGENTS, HORIZON_TO_TEST, N_TRIALS, SEED,
                    SCAN_N_POINTS, SCAN_N_REPEATS,
                    MCMC_N_WALKERS, MCMC_N_BURNIN, MCMC_N_STEPS,
                    MCMC_SCAN_N_POINTS, MCMC_SCAN_N_STEPS, MCMC_SCAN_N_BURNIN,
                    COMPARISON_N_AGENTS, COMPARISON_N_REPEATS,
                    COMPARISON_N_STARTS, COMPARISON_MODELS)
from paths import get_output_dir

from analysis.simulation import generate_data
from analysis.fitting import fit_model, compute_landscape_data
from analysis.recovery import run_parameter_recovery, run_distributional_recovery, compute_recovery_errors
from analysis.psp import run_psp_analysis, summarize_psp
from analysis.diagnostics import run_repeated_diagnostics, check_sensitivity, run_reliability_scan
from analysis.mcmc import run_mcmc, compute_mcmc_diagnostics, print_mcmc_diagnostics, run_mcmc_reliability_scan
from analysis.behavioral import SteingroeverStrategy
from analysis.comparison import run_cross_fit_comparison, summarize_cross_fit

from plots.recovery import plot_recovery_results, plot_recovery_thesis
from plots.diagnostics import plot_full_identification_matrix, plot_full_diagnostic_matrix, plot_parameter_error_profiles
from plots.psp import plot_psp_results, plot_psp_distribution
from plots.gsa import (plot_gsa_marginal, plot_gsa_coupling, plot_all_error_landscapes,
                       plot_error_normality, plot_reliability_map,
                       plot_reliability_parallel_coords, plot_reliability_slice_heatmaps,
                       plot_reliability_quadrant, plot_reliability_bias_profiles)
from plots.mcmc import plot_mcmc_traces, plot_mcmc_corner, plot_mcmc_summary
from plots.mcmc_scan import (plot_mcmc_scan_kl, plot_mcmc_scan_corr_heatmap,
                              plot_mcmc_scan_summary, plot_mcmc_scan_landscapes)
from plots.comparison import plot_cross_fit_results


def run_main():
    config = AGENT_CONFIGS[ACTIVE_AGENT]

    # COMPARISON mode uses its own output dir and ignores ACTIVE_AGENT
    if MODE == "COMPARISON":
        output_dir = get_output_dir("comparison", "comparison")
    else:
        output_dir = get_output_dir(config.name, MODE.lower())

    print(f"Agent:  {config.name if MODE != 'COMPARISON' else '(all models)'}")
    print(f"Mode:   {MODE}")
    print(f"Output: {output_dir}")
    print()

    # ------------------------------------------------------------------
    if MODE == "RECOVERY":
        if not REPLOT:
            result = run_parameter_recovery(config, N_AGENTS, HORIZON_TO_TEST, N_TRIALS, SEED)
            if result is None:
                print("All fits failed — nothing to plot.")
                return
            df, _, _ = result
            df = compute_recovery_errors(df, config)
            df.to_csv(output_dir / "recovery_diagnostics.csv", index=False)

        df = pd.read_csv(output_dir / "recovery_diagnostics.csv")
        plot_recovery_results(df, HORIZON_TO_TEST, N_TRIALS,
                              config.param_names, config.bounds, output_dir)
        plot_recovery_thesis(df, config.param_names, config.bounds, output_dir, model_name=config.name)
        plot_gsa_marginal(df, config.param_names, output_dir)
        plot_gsa_coupling(df, config.param_names, output_dir)
        plot_all_error_landscapes(df, config.param_names, output_dir)

    # ------------------------------------------------------------------
    elif MODE == "DIST_RECOVERY":
        if not REPLOT:
            df = run_distributional_recovery(config, n_agents=50, n_repeats=50,
                                             horizon=HORIZON_TO_TEST, seed=SEED)
            df.to_csv(output_dir / "distributional_recovery.csv", index=False)

        df = pd.read_csv(output_dir / "distributional_recovery.csv")
        plot_error_normality(df, config.param_names, output_dir)

    # ------------------------------------------------------------------
    elif MODE == "PSP":
        broad = SteingroeverStrategy(restricted=False)
        restr = SteingroeverStrategy(restricted=True)

        for strat in (broad, restr):
            csv_path = output_dir / f"psp_{strat.name.lower()}.csv"
            if not REPLOT:
                print(f"Running PSP ({strat.name})...")
                df = run_psp_analysis(5000, config, N_TRIALS, HORIZON_TO_TEST, strat)
                df.to_csv(csv_path, index=False)

            df = pd.read_csv(csv_path)
            _, percents = summarize_psp(df, strat.name)
            plot_psp_results(df, strat.name, output_dir)
            plot_psp_distribution(percents, strat.name, output_dir)

    # ------------------------------------------------------------------
    elif MODE == "FULL_LANDSCAPE":
        test_p = config.default_test_params
        print(f"Test params: {dict(zip(config.param_names, test_p))}")

        npz_path = output_dir / "landscape.npz"
        if not REPLOT:
            acts, outs = generate_data(test_p, config=config,
                                       n_trials=N_TRIALS, horizon=HORIZON_TO_TEST, seed=SEED)
            mle_p = fit_model(acts, outs, config=config,
                              horizon=HORIZON_TO_TEST, method="MLE", seed=SEED)
            if mle_p is None:
                print("Fitting failed — cannot generate landscape.")
                return
            print("Computing NLL landscape grids...")
            landscape = compute_landscape_data(test_p, mle_p, acts, outs, config, HORIZON_TO_TEST)
            np.savez(npz_path, **landscape)
            print(f"Saved: {npz_path}")
            check_sensitivity(mle_p, acts, outs, config, HORIZON_TO_TEST)

        plot_full_identification_matrix(npz_path, config.param_names, output_dir,
                                        model_name=config.name)

    # ------------------------------------------------------------------
    elif MODE == "DIAGNOSTICS":
        test_p = config.default_test_params
        print(f"Test params: {dict(zip(config.param_names, test_p))}")

        if not REPLOT:
            est_samples = run_repeated_diagnostics(test_p, N_AGENTS, config, HORIZON_TO_TEST)
            pd.DataFrame(est_samples, columns=config.param_names).to_csv(
                output_dir / "diagnostics_estimates.csv", index=False
            )
            pd.DataFrame([test_p], columns=config.param_names).to_csv(
                output_dir / "diagnostics_true.csv", index=False
            )

        df_est   = pd.read_csv(output_dir / "diagnostics_estimates.csv")
        true_p   = pd.read_csv(output_dir / "diagnostics_true.csv").iloc[0].values
        plot_full_diagnostic_matrix(df_est, true_p, config.param_names, config.bounds, output_dir)
        plot_parameter_error_profiles(df_est, true_p, config.param_names, config.bounds, output_dir)

    # ------------------------------------------------------------------
    elif MODE == "RELIABILITY_SCAN":
        print(f"Scan: {SCAN_N_POINTS} points × {SCAN_N_REPEATS} repeats "
              f"= {SCAN_N_POINTS * SCAN_N_REPEATS} total fits")
        if not REPLOT:
            df_scan = run_reliability_scan(
                config, SCAN_N_POINTS, SCAN_N_REPEATS, HORIZON_TO_TEST, SEED
            )
            df_scan.to_csv(output_dir / "reliability_scan.csv", index=False)
            print(f"Saved: {output_dir / 'reliability_scan.csv'}")

        df_scan = pd.read_csv(output_dir / "reliability_scan.csv")
        THRESHOLD = 0.2
        plot_reliability_quadrant(df_scan, output_dir, THRESHOLD)
        plot_reliability_map(df_scan, config.param_names, output_dir)
        plot_reliability_parallel_coords(df_scan, config.param_names,
                                         config.bounds, output_dir, THRESHOLD)
        plot_reliability_slice_heatmaps(df_scan, config.param_names,
                                        config.bounds, output_dir, THRESHOLD)
        plot_reliability_bias_profiles(df_scan, config.param_names,
                                       config.bounds, output_dir)

        n_safe = (df_scan["rmsse_median"] < THRESHOLD).sum()
        print(f"\nSafe points (RMSSE < {THRESHOLD}): {n_safe} / {len(df_scan)} "
              f"({n_safe/len(df_scan)*100:.1f}%)")
        print(f"Points with ≥80% normal errors: "
              f"{(df_scan['pct_normal'] >= 80).sum()} / {len(df_scan)}")

    # ------------------------------------------------------------------
    elif MODE == "MCMC":
        test_p = config.default_test_params
        print(f"Test params: {dict(zip(config.param_names, test_p))}")
        print(f"Walkers: {MCMC_N_WALKERS}  Burn-in: {MCMC_N_BURNIN}  Steps: {MCMC_N_STEPS}")

        if not REPLOT:
            acts, outs = generate_data(test_p, config=config,
                                       n_trials=N_TRIALS, horizon=HORIZON_TO_TEST, seed=SEED)
            sampler, flat_samples, chain = run_mcmc(
                acts, outs, config,
                horizon=HORIZON_TO_TEST,
                n_walkers=MCMC_N_WALKERS,
                n_steps=MCMC_N_STEPS,
                n_burnin=MCMC_N_BURNIN,
                seed=SEED,
            )
            diags = compute_mcmc_diagnostics(sampler, flat_samples, config)
            print_mcmc_diagnostics(diags)

            np.save(output_dir / "mcmc_samples.npy", flat_samples)
            np.save(output_dir / "mcmc_chain.npy", chain)
            diags_records = [
                {
                    "param":         name,
                    "kl_div":        d["kl_div"],
                    "max_post_corr": d["max_post_corr"],
                    "mean":          d["mean"],
                    "std":           d["std"],
                    "ci_lo":         d["ci_95"][0],
                    "ci_hi":         d["ci_95"][1],
                    "shrinkage":     d["shrinkage"],
                    "ess":           d["ess"] if d["ess"] is not None else float("nan"),
                }
                for name, d in diags.items()
            ]
            pd.DataFrame(diags_records).to_csv(output_dir / "mcmc_diags.csv", index=False)
            pd.DataFrame([test_p], columns=config.param_names).to_csv(
                output_dir / "mcmc_true_params.csv", index=False
            )

        chain       = np.load(output_dir / "mcmc_chain.npy")
        flat_samples = np.load(output_dir / "mcmc_samples.npy")
        df_diags    = pd.read_csv(output_dir / "mcmc_diags.csv")
        true_p      = pd.read_csv(output_dir / "mcmc_true_params.csv").iloc[0].values

        plot_mcmc_traces(chain, config.param_names, output_dir)
        plot_mcmc_corner(flat_samples, config.param_names, config.bounds, true_p, output_dir)
        plot_mcmc_summary(df_diags, config.param_names, config.bounds, output_dir)

    # ------------------------------------------------------------------
    elif MODE == "COMPARISON":
        configs_to_compare = [AGENT_CONFIGS[k] for k in COMPARISON_MODELS]

        print(f"Models:  {[c.name for c in configs_to_compare]}")
        print(f"Agents:  {COMPARISON_N_AGENTS} per model")
        print(f"Repeats: {COMPARISON_N_REPEATS} per agent per fit-model")
        print(f"Starts:  {COMPARISON_N_STARTS} L-BFGS-B starts per fit")
        total = (COMPARISON_N_AGENTS
                 * len(configs_to_compare) ** 2
                 * COMPARISON_N_REPEATS)
        print(f"Total fits: ~{total}")
        print()

        csv_path = output_dir / "cross_fit_raw.csv"
        if not REPLOT:
            df = run_cross_fit_comparison(
                configs=configs_to_compare,
                n_agents=COMPARISON_N_AGENTS,
                n_trials=N_TRIALS,
                n_repeats=COMPARISON_N_REPEATS,
                n_fit_starts=COMPARISON_N_STARTS,
                seed=SEED,
            )
            df.to_csv(csv_path, index=False)
            print(f"Raw results: {csv_path}")

        df = pd.read_csv(csv_path)
        summary = summarize_cross_fit(df)
        summary.to_csv(output_dir / "cross_fit_summary.csv", index=False)
        print("\n=== CROSS-FIT SUMMARY ===")
        print(summary.to_string(index=False))

        plot_cross_fit_results(df, output_dir)

    # ------------------------------------------------------------------
    elif MODE == "MCMC_SCAN":
        print(f"MCMC scan: {MCMC_SCAN_N_POINTS} points  "
              f"{MCMC_N_WALKERS} walkers × {MCMC_SCAN_N_STEPS} steps")

        if not REPLOT:
            df_scan = run_mcmc_reliability_scan(
                config=config,
                n_points=MCMC_SCAN_N_POINTS,
                horizon=HORIZON_TO_TEST,
                seed=SEED,
                n_walkers=MCMC_N_WALKERS,
                n_steps=MCMC_SCAN_N_STEPS,
                n_burnin=MCMC_SCAN_N_BURNIN,
            )
            df_scan.to_csv(output_dir / "mcmc_scan.csv", index=False)
            print(f"Saved: {output_dir / 'mcmc_scan.csv'}")

        df_scan = pd.read_csv(output_dir / "mcmc_scan.csv")
        n_identified = (df_scan["mean_kl_div"]   >= 1.0).sum()
        n_confounded = (df_scan["max_post_corr"] >= 0.7).sum()
        print(f"\nIdentified (mean KL ≥ 1.0 nat): {n_identified} / {len(df_scan)}")
        print(f"Confounded (max post. corr ≥ 0.7): {n_confounded} / {len(df_scan)}")

        plot_mcmc_scan_summary(df_scan, config.param_names, output_dir)
        plot_mcmc_scan_kl(df_scan, config.param_names, output_dir)
        plot_mcmc_scan_corr_heatmap(df_scan, config.param_names, output_dir)
        plot_mcmc_scan_landscapes(df_scan, config.param_names, config.bounds, output_dir)

    # ------------------------------------------------------------------
    else:
        print(f"Unknown mode: {MODE!r}")
        print(f"Valid modes: RECOVERY, DIST_RECOVERY, PSP, FULL_LANDSCAPE, "
              f"DIAGNOSTICS, RELIABILITY_SCAN, MCMC, MCMC_SCAN, COMPARISON")


if __name__ == "__main__":
    run_main()