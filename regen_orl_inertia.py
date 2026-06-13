"""Zmiana etykiety Perseverance -> Inertia dla ORL i regeneracja wykresów."""
import matplotlib
matplotlib.use("Agg")
import sys, shutil
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
from pathlib import Path
from config import AGENT_CONFIGS

cfg = AGENT_CONFIGS["ORLAgent"]
base = Path("results/ORL")
figures = Path("../praca_magisterska/figures")

OLD = r"$\beta_P$ (Perseverance)"
NEW = r"$\beta_P$ (Inertia)"

def rename_csv(path):
    if not path.exists():
        print(f"  BRAK: {path}"); return
    df = pd.read_csv(path)
    rename_map = {col: col.replace(OLD, NEW) for col in df.columns if OLD in col}
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        df.to_csv(path, index=False)
        print(f"  Przemianowano {len(rename_map)} kolumn: {path.name}")
    else:
        print(f"  Brak zmian: {path.name}")

def rename_index_col(path):
    if not path.exists(): return
    df = pd.read_csv(path)
    if "param" in df.columns:
        df["param"] = df["param"].str.replace(OLD, NEW, regex=False)
        df.to_csv(path, index=False)
        print(f"  Przemianowano indeks: {path.name}")

def newest(directory, pattern):
    files = list(directory.glob(pattern))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None

def copy_newest(src_dir, pattern, dest):
    f = newest(src_dir, pattern)
    if f: shutil.copy(f, dest); print(f"  -> {dest.name}")

# --- 1. Przemianuj CSV ---
print("=== Przemianowanie CSV ===")
for p in [
    base / "reliability_scan/reliability_scan.csv",
    base / "mcmc_scan/mcmc_scan.csv",
    base / "recovery/recovery_diagnostics.csv",
]:
    rename_csv(p)
rename_index_col(base / "mcmc/mcmc_diags.csv")

# --- 2. Regeneruj wykresy ---
print("\n=== Regeneracja wykresów ===")
from plots.recovery import plot_recovery_thesis
from plots.gsa import (plot_reliability_map, plot_reliability_slice_heatmaps,
                        plot_reliability_bias_profiles, plot_reliability_quadrant)
from plots.mcmc import plot_mcmc_corner, plot_mcmc_summary
from plots.mcmc_scan import (plot_mcmc_scan_summary, plot_mcmc_scan_corr_heatmap,
                              plot_mcmc_scan_landscapes)

THRESHOLD = 0.2

# Recovery
print("Recovery...")
rec_dir = base / "recovery"
if (rec_dir / "recovery_diagnostics.csv").exists():
    df_rec = pd.read_csv(rec_dir / "recovery_diagnostics.csv")
    plot_recovery_thesis(df_rec, cfg.param_names, cfg.bounds, rec_dir, model_name="ORL")
    copy_newest(rec_dir, "recovery_thesis_*.png", figures / "parameter_recovery/ORL.png")

# Reliability scan
print("Reliability scan...")
rel_dir = base / "reliability_scan"
df_rel = pd.read_csv(rel_dir / "reliability_scan.csv")
plot_reliability_quadrant(df_rel, rel_dir, THRESHOLD)
plot_reliability_map(df_rel, cfg.param_names, rel_dir)
plot_reliability_slice_heatmaps(df_rel, cfg.param_names, cfg.bounds, rel_dir)
plot_reliability_bias_profiles(df_rel, cfg.param_names, cfg.bounds, rel_dir)
copy_newest(rel_dir, "reliability_quadrant_*.png",     figures / "reliability_scan/orl_quadrant.png")
copy_newest(rel_dir, "reliability_rmsse_*.png",         figures / "reliability_scan/orl_rmsse.png")
copy_newest(rel_dir, "reliability_slices_*.png",        figures / "reliability_scan/orl_slices.png")
copy_newest(rel_dir, "reliability_bias_profiles_*.png", figures / "reliability_scan/orl_bias_profiles.png")

# MCMC single-point
print("MCMC single-point...")
mcmc_dir = base / "mcmc"
if (mcmc_dir / "mcmc_samples.npy").exists():
    flat_samples = np.load(mcmc_dir / "mcmc_samples.npy")
    df_diags = pd.read_csv(mcmc_dir / "mcmc_diags.csv")
    true_p   = pd.read_csv(mcmc_dir / "mcmc_true_params.csv").iloc[0].values
    plot_mcmc_corner(flat_samples, cfg.param_names, cfg.bounds, true_p, mcmc_dir)
    plot_mcmc_summary(df_diags, cfg.param_names, cfg.bounds, mcmc_dir)
    copy_newest(mcmc_dir, "mcmc_corner_*.png",  figures / "mcmc/orl_corner.png")
    copy_newest(mcmc_dir, "mcmc_summary_*.png", figures / "mcmc/orl_summary.png")

# MCMC scan
print("MCMC scan...")
scan_dir = base / "mcmc_scan"
df_scan = pd.read_csv(scan_dir / "mcmc_scan.csv")
plot_mcmc_scan_summary(df_scan, cfg.param_names, scan_dir)
plot_mcmc_scan_corr_heatmap(df_scan, cfg.param_names, scan_dir)
plot_mcmc_scan_landscapes(df_scan, cfg.param_names, cfg.bounds, scan_dir)
copy_newest(scan_dir, "mcmc_scan_summary_*.png",      figures / "mcmc_scan/orl_scan_summary.png")
copy_newest(scan_dir, "mcmc_scan_corr_heatmap_*.png", figures / "mcmc_scan/orl_scan_corr.png")
copy_newest(scan_dir, "mcmc_scan_landscapes_*.png",   figures / "mcmc_scan/orl_safe_map.png")

print("\nGotowe.")
