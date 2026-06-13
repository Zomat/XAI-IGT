"""Regeneruje wykresy MCMC i MCMC_SCAN dla wszystkich modeli z istniejących danych."""
import matplotlib
matplotlib.use("Agg")
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from config import AGENT_CONFIGS
from paths import get_output_dir
from plots.mcmc import plot_mcmc_traces, plot_mcmc_corner, plot_mcmc_summary
from plots.mcmc_scan import (plot_mcmc_scan_kl, plot_mcmc_scan_corr_heatmap,
                              plot_mcmc_scan_summary, plot_mcmc_scan_landscapes)

AGENTS = ["Agent", "PVLDeltaAgent", "ORLAgent"]

for key in AGENTS:
    cfg = AGENT_CONFIGS[key]

    # MCMC single-point
    mcmc_dir = get_output_dir(cfg.name, "mcmc")
    samples_f = mcmc_dir / "mcmc_samples.npy"
    chain_f   = mcmc_dir / "mcmc_chain.npy"
    diags_f   = mcmc_dir / "mcmc_diags.csv"
    true_f    = mcmc_dir / "mcmc_true_params.csv"
    if samples_f.exists() and chain_f.exists():
        print(f"Regenerating MCMC plots for {cfg.name}...")
        flat_samples = np.load(samples_f)
        chain        = np.load(chain_f)
        df_diags     = pd.read_csv(diags_f)
        true_p       = pd.read_csv(true_f).iloc[0].values
        plot_mcmc_traces(chain, cfg.param_names, mcmc_dir)
        plot_mcmc_corner(flat_samples, cfg.param_names, cfg.bounds, true_p, mcmc_dir)
        plot_mcmc_summary(df_diags, cfg.param_names, cfg.bounds, mcmc_dir)
        print(f"  Done: {cfg.name} MCMC")
    else:
        print(f"  Skipping {cfg.name} MCMC — no data")

    # MCMC_SCAN
    scan_dir = get_output_dir(cfg.name, "mcmc_scan")
    scan_f   = scan_dir / "mcmc_scan.csv"
    if scan_f.exists():
        print(f"Regenerating MCMC_SCAN plots for {cfg.name}...")
        df_scan = pd.read_csv(scan_f)
        # CSV may predate parameter renames in config — use names stored in the CSV
        csv_names = [c[len("kl_div_"):] for c in df_scan.columns
                     if c.startswith("kl_div_") and c != "mean_kl_div"]
        plot_mcmc_scan_summary(df_scan, csv_names, scan_dir)
        plot_mcmc_scan_kl(df_scan, csv_names, scan_dir)
        plot_mcmc_scan_corr_heatmap(df_scan, csv_names, scan_dir)
        plot_mcmc_scan_landscapes(df_scan, csv_names, cfg.bounds, scan_dir)
        print(f"  Done: {cfg.name} MCMC_SCAN")
    else:
        print(f"  Skipping {cfg.name} MCMC_SCAN — no data")

print("Wszystkie wykresy MCMC zregenerowane.")
