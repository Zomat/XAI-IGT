"""Przemianowuje kolumny CSV dla PVL-Delta (alpha->A, lambda->w) i regeneruje wykresy."""
import matplotlib
matplotlib.use("Agg")
import sys, shutil
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
from pathlib import Path
from config import AGENT_CONFIGS

cfg = AGENT_CONFIGS["PVLDeltaAgent"]
base = Path("results/PVL-Delta")
figures = Path("../praca_magisterska/figures")

OLD_NEW = {
    r"$\alpha$ (Shape)":          "A (Shape)",
    r"$\lambda$ (Loss Aversion)": "w (Loss Aversion)",
    r"$a$ (Learning Rate)":       "a (Learning Rate)",
    r"$c$ (Consistency)":         "c (Consistency)",
}

def rename_csv(path):
    if not path.exists():
        print(f"  BRAK: {path}"); return
    df = pd.read_csv(path)
    rename_map = {col: col.replace(old, new)
                  for col in df.columns
                  for old, new in OLD_NEW.items() if old in col}
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        df.to_csv(path, index=False)
        print(f"  Przemianowano {len(rename_map)} kolumn: {path.name}")
    else:
        print(f"  Brak zmian: {path.name}")

def newest(directory, pattern):
    files = list(directory.glob(pattern))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None

def copy_newest(src_dir, pattern, dest):
    f = newest(src_dir, pattern)
    if f:
        shutil.copy(f, dest)
        print(f"  -> {dest.name}")

# --- 1. Przemianuj kolumny w CSV ---
print("=== Przemianowanie CSV ===")
for csv_path in [
    base / "reliability_scan/reliability_scan.csv",
    base / "psp/psp_broad.csv",
    base / "psp/psp_restricted.csv",
    base / "mcmc_scan/mcmc_scan.csv",
]:
    rename_csv(csv_path)

# mcmc_diags.csv ma stare nazwy w kolumnie 'param' (indeks)
diags_path = base / "mcmc/mcmc_diags.csv"
if diags_path.exists():
    df_d = pd.read_csv(diags_path)
    df_d["param"] = df_d["param"].replace({old: new for old, new in OLD_NEW.items()})
    df_d.to_csv(diags_path, index=False)
    print(f"  Przemianowano indeks: {diags_path.name}")

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
df_rec = pd.read_csv(rec_dir / "recovery_diagnostics.csv")
plot_recovery_thesis(df_rec, cfg.param_names, cfg.bounds, rec_dir, model_name="PVL-Delta")
copy_newest(rec_dir, "recovery_thesis_*.png", figures / "parameter_recovery/pvl_delta.png")

# Reliability scan
print("Reliability scan...")
rel_dir = base / "reliability_scan"
df_rel = pd.read_csv(rel_dir / "reliability_scan.csv")
plot_reliability_quadrant(df_rel, rel_dir, THRESHOLD)
plot_reliability_map(df_rel, cfg.param_names, rel_dir)
plot_reliability_slice_heatmaps(df_rel, cfg.param_names, cfg.bounds, rel_dir)
plot_reliability_bias_profiles(df_rel, cfg.param_names, cfg.bounds, rel_dir)
copy_newest(rel_dir, "reliability_quadrant_*.png",     figures / "reliability_scan/pvl_quadrant.png")
copy_newest(rel_dir, "reliability_rmsse_*.png",         figures / "reliability_scan/pvl_rmsse.png")
copy_newest(rel_dir, "reliability_slices_*.png",        figures / "reliability_scan/pvl_slices.png")
copy_newest(rel_dir, "reliability_bias_profiles_*.png", figures / "reliability_scan/pvl_bias_profiles.png")

# MCMC single-point
print("MCMC single-point...")
mcmc_dir = base / "mcmc"
if (mcmc_dir / "mcmc_samples.npy").exists():
    flat_samples = np.load(mcmc_dir / "mcmc_samples.npy")
    df_diags = pd.read_csv(mcmc_dir / "mcmc_diags.csv")
    true_p   = pd.read_csv(mcmc_dir / "mcmc_true_params.csv").iloc[0].values
    plot_mcmc_corner(flat_samples, cfg.param_names, cfg.bounds, true_p, mcmc_dir)
    plot_mcmc_summary(df_diags, cfg.param_names, cfg.bounds, mcmc_dir)
    copy_newest(mcmc_dir, "mcmc_corner_*.png",  figures / "mcmc/pvl_corner.png")
    copy_newest(mcmc_dir, "mcmc_summary_*.png", figures / "mcmc/pvl_summary.png")

# MCMC scan
print("MCMC scan...")
scan_dir = base / "mcmc_scan"
df_scan = pd.read_csv(scan_dir / "mcmc_scan.csv")
plot_mcmc_scan_summary(df_scan, cfg.param_names, scan_dir)
plot_mcmc_scan_corr_heatmap(df_scan, cfg.param_names, scan_dir)
plot_mcmc_scan_landscapes(df_scan, cfg.param_names, cfg.bounds, scan_dir)
copy_newest(scan_dir, "mcmc_scan_summary_*.png",      figures / "mcmc_scan/pvl_delta_scan_summary.png")
copy_newest(scan_dir, "mcmc_scan_corr_heatmap_*.png", figures / "mcmc_scan/pvl_delta_scan_corr.png")
copy_newest(scan_dir, "mcmc_scan_landscapes_*.png",   figures / "mcmc_scan/pvl_safe_map.png")

print("\nGotowe.")
