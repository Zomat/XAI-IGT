"""Regeneruje tylko corner ploty z istniejących danych npy."""
import matplotlib
matplotlib.use("Agg")
import sys, shutil
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path
from config import AGENT_CONFIGS
from plots.mcmc import plot_mcmc_corner

FIGURES = Path("../praca_magisterska/figures/mcmc")
NAMES = {
    "Agent":        "kpvl_corner.png",
    "PVLDeltaAgent": "pvl_corner.png",
    "ORLAgent":     "orl_corner.png",
}

for key, fname in NAMES.items():
    cfg = AGENT_CONFIGS[key]
    mcmc_dir = Path("results") / cfg.name / "mcmc"
    samples_f   = mcmc_dir / "mcmc_samples.npy"
    true_f      = mcmc_dir / "mcmc_true_params.csv"
    if not samples_f.exists():
        print(f"BRAK danych: {samples_f}")
        continue

    flat_samples = np.load(samples_f)
    true_p       = pd.read_csv(true_f).iloc[0].values
    print(f"Generuję corner dla {cfg.name}...")
    plot_mcmc_corner(flat_samples, cfg.param_names, cfg.bounds, true_p, mcmc_dir)

    newest = max(mcmc_dir.glob("mcmc_corner_*.png"), key=lambda p: p.stat().st_mtime)
    dest = FIGURES / fname
    shutil.copy(newest, dest)
    print(f"  -> {dest}")

print("Gotowe.")
