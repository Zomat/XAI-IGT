"""
Batch MCMC scan — 500 punktów LHS dla KPVL, PVL-Delta, ORL.
Uruchom: python batch_mcmc_scan_500.py  (z aktywowanym venv)
Logi:    logs/batch_mcmc_scan_500.log
"""
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import AGENT_CONFIGS, MCMC_N_WALKERS, SEED, HORIZON_TO_TEST
from analysis.mcmc import run_mcmc_reliability_scan
from paths import get_output_dir
from plots.mcmc_scan import (
    plot_mcmc_scan_summary, plot_mcmc_scan_kl,
    plot_mcmc_scan_corr_heatmap, plot_mcmc_scan_landscapes,
    plot_mcmc_scan_safe_map,
)

# ── settings ──────────────────────────────────────────────────────────────────
N_POINTS  = 500
N_STEPS   = 1000
N_BURNIN  = 300
AGENTS    = ["Agent", "PVLDeltaAgent", "ORLAgent"]

# ── logging ───────────────────────────────────────────────────────────────────
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_path = log_dir / "batch_mcmc_scan_500.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger()

# ── main ──────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info(f"Batch MCMC scan 500 pkt — start  ({datetime.now():%Y-%m-%d %H:%M})")
log.info(f"  {N_POINTS} pkt × {MCMC_N_WALKERS} walkerów × ({N_BURNIN} burnin + {N_STEPS} kroków)")
log.info("=" * 60)

for i, agent_key in enumerate(AGENTS, 1):
    cfg = AGENT_CONFIGS[agent_key]
    output_dir = get_output_dir(cfg.name, "mcmc_scan")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n[{i}/{len(AGENTS)}] MCMC_SCAN → model {cfg.name}")
    t0 = time.time()

    df = run_mcmc_reliability_scan(
        config=cfg,
        n_points=N_POINTS,
        horizon=HORIZON_TO_TEST,
        seed=SEED,
        n_walkers=MCMC_N_WALKERS,
        n_steps=N_STEPS,
        n_burnin=N_BURNIN,
    )
    df.to_csv(output_dir / "mcmc_scan.csv", index=False)
    log.info(f"  Zapisano: {output_dir / 'mcmc_scan.csv'}")

    # summary stats
    if "mean_kl_div" in df.columns:
        n_id = (df["mean_kl_div"] >= 1.0).sum()
        log.info(f"  Zidentyfikowane (KL ≥ 1.0): {n_id}/{N_POINTS}")
    if "max_post_corr" in df.columns:
        n_conf = (df["max_post_corr"] >= 0.7).sum()
        log.info(f"  Skonfundowane (corr ≥ 0.7): {n_conf}/{N_POINTS}")

    # plots
    log.info("  Generowanie wykresów...")
    param_names = [c.replace("true_", "") for c in df.columns if c.startswith("true_")]
    plot_mcmc_scan_summary(df, param_names, output_dir)
    plot_mcmc_scan_kl(df, param_names, output_dir)
    plot_mcmc_scan_corr_heatmap(df, param_names, output_dir)
    plot_mcmc_scan_landscapes(df, param_names, cfg.bounds, output_dir)
    plot_mcmc_scan_safe_map(df, param_names, output_dir)

    elapsed = (time.time() - t0) / 60
    log.info(f"  MCMC_SCAN/{cfg.name}: gotowe.  Czas: {elapsed:.1f} min")

log.info("\n" + "=" * 60)
log.info("Wszystkie skany zakończone.")
log.info("=" * 60)
