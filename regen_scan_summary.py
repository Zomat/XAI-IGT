"""Regeneruje tylko scan_summary wykresy (KL + ESS, bez SD) z istniejących CSV."""
import matplotlib
matplotlib.use("Agg")
import sys, shutil
sys.path.insert(0, "src")

import pandas as pd
from pathlib import Path
from config import AGENT_CONFIGS
from plots.mcmc_scan import plot_mcmc_scan_summary

FIGURES = {
    "Agent":         Path("../praca_magisterska/figures/mcmc_scan/kpvl_scan_summary.png"),
    "PVLDeltaAgent": Path("../praca_magisterska/figures/mcmc_scan/pvl_delta_scan_summary.png"),
    "ORLAgent":      Path("../praca_magisterska/figures/mcmc_scan/orl_scan_summary.png"),
}

for key, dest in FIGURES.items():
    cfg = AGENT_CONFIGS[key]
    scan_dir = Path("results") / cfg.name / "mcmc_scan"
    scan_f   = scan_dir / "mcmc_scan.csv"
    if not scan_f.exists():
        print(f"BRAK danych: {scan_f}")
        continue
    print(f"Generuję scan_summary dla {cfg.name}...")
    df = pd.read_csv(scan_f)
    plot_mcmc_scan_summary(df, cfg.param_names, scan_dir)
    newest = max(scan_dir.glob("mcmc_scan_summary_*.png"), key=lambda p: p.stat().st_mtime)
    shutil.copy(newest, dest)
    print(f"  -> {dest}")

print("Gotowe.")
