"""
Batch runner dla analiz MCMC i MCMC_SCAN dla wszystkich modeli.
Uruchamia sekwencyjnie bez modyfikacji config.py.

Użycie:
    cd /home/mateusz/projects/igt_sims
    source igt_env/bin/activate
    nohup python src/batch_mcmc.py > logs/batch_mcmc.log 2>&1 &
"""

import matplotlib
matplotlib.use("Agg")

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from config import AGENT_CONFIGS
from paths import get_output_dir

from analysis.simulation import generate_data
from analysis.fitting import fit_model
from analysis.mcmc import (run_mcmc, compute_mcmc_diagnostics,
                            print_mcmc_diagnostics, run_mcmc_reliability_scan)
from plots.mcmc import plot_mcmc_traces, plot_mcmc_corner, plot_mcmc_summary
from plots.mcmc_scan import (plot_mcmc_scan_kl, plot_mcmc_scan_corr_heatmap,
                              plot_mcmc_scan_summary, plot_mcmc_scan_landscapes)

# ── wspólne ustawienia ────────────────────────────────────────────────────────
N_TRIALS        = 100
HORIZON         = 0
SEED            = 42

MCMC_N_WALKERS  = 32
MCMC_N_BURNIN   = 500
MCMC_N_STEPS    = 2000

MCMC_SCAN_N_POINTS = 100
MCMC_SCAN_N_STEPS  = 1000
MCMC_SCAN_N_BURNIN = 300

# ── kolejka zadań ─────────────────────────────────────────────────────────────
# (klucz AGENT_CONFIGS, tryb, czy pomijać jeśli CSV już istnieje)
JOBS = [
    ("Agent",         "MCMC_SCAN"),   # KPVL   — brak
    ("PVLDeltaAgent", "MCMC"),        # PVL-Delta — brak
    ("PVLDeltaAgent", "MCMC_SCAN"),   # PVL-Delta — brak
    ("ORLAgent",      "MCMC"),        # ORL    — brak
    ("ORLAgent",      "MCMC_SCAN"),   # ORL    — brak
]

# ─────────────────────────────────────────────────────────────────────────────

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_mcmc_job(agent_key, cfg):
    output_dir = get_output_dir(cfg.name, "mcmc")
    csv_path   = output_dir / "mcmc_diags.csv"

    if csv_path.exists():
        log(f"  MCMC/{cfg.name}: CSV już istnieje — pomijam obliczenia, generuję wykresy.")
    else:
        test_p = cfg.default_test_params
        log(f"  Generowanie danych: {dict(zip(cfg.param_names, test_p))}")
        acts, outs = generate_data(test_p, config=cfg,
                                   n_trials=N_TRIALS, horizon=HORIZON, seed=SEED)

        log(f"  MLE fit...")
        mle_p = fit_model(acts, outs, config=cfg, horizon=HORIZON,
                           method="MLE", seed=SEED)
        if mle_p is None:
            log("  MLE nie zeszło — używam default_test_params jako start walkerów.")

        log(f"  Uruchamiam MCMC ({MCMC_N_WALKERS} walkerów, "
            f"{MCMC_N_BURNIN} burnin + {MCMC_N_STEPS} kroków)...")
        sampler, flat_samples, chain = run_mcmc(
            acts, outs, cfg,
            horizon=HORIZON,
            n_walkers=MCMC_N_WALKERS,
            n_steps=MCMC_N_STEPS,
            n_burnin=MCMC_N_BURNIN,
            seed=SEED,
        )
        diags = compute_mcmc_diagnostics(sampler, flat_samples, cfg)
        print_mcmc_diagnostics(diags)

        np.save(output_dir / "mcmc_samples.npy", flat_samples)
        np.save(output_dir / "mcmc_chain.npy", chain)

        records = [
            {"param": name, "kl_div": d["kl_div"], "max_post_corr": d["max_post_corr"],
             "mean": d["mean"], "std": d["std"],
             "ci_lo": d["ci_95"][0], "ci_hi": d["ci_95"][1],
             "shrinkage": d["shrinkage"],
             "ess": d["ess"] if d["ess"] is not None else float("nan")}
            for name, d in diags.items()
        ]
        pd.DataFrame(records).to_csv(csv_path, index=False)
        pd.DataFrame([test_p], columns=cfg.param_names).to_csv(
            output_dir / "mcmc_true_params.csv", index=False)
        log(f"  Zapisano: {csv_path}")

    log("  Generowanie wykresów MCMC...")
    chain        = np.load(output_dir / "mcmc_chain.npy")
    flat_samples = np.load(output_dir / "mcmc_samples.npy")
    df_diags     = pd.read_csv(output_dir / "mcmc_diags.csv")
    true_p       = pd.read_csv(output_dir / "mcmc_true_params.csv").iloc[0].values
    plot_mcmc_traces(chain, cfg.param_names, output_dir)
    plot_mcmc_corner(flat_samples, cfg.param_names, cfg.bounds, true_p, output_dir)
    plot_mcmc_summary(df_diags, cfg.param_names, cfg.bounds, output_dir)
    log(f"  MCMC/{cfg.name}: gotowe.")


def run_mcmc_scan_job(agent_key, cfg):
    output_dir = get_output_dir(cfg.name, "mcmc_scan")
    csv_path   = output_dir / "mcmc_scan.csv"

    if csv_path.exists():
        log(f"  MCMC_SCAN/{cfg.name}: CSV już istnieje — pomijam obliczenia, generuję wykresy.")
    else:
        log(f"  Uruchamiam MCMC_SCAN: {MCMC_SCAN_N_POINTS} punktów × "
            f"{MCMC_N_WALKERS} walkerów × "
            f"({MCMC_SCAN_N_BURNIN} burnin + {MCMC_SCAN_N_STEPS} kroków) ...")
        df_scan = run_mcmc_reliability_scan(
            config=cfg,
            n_points=MCMC_SCAN_N_POINTS,
            horizon=HORIZON,
            seed=SEED,
            n_walkers=MCMC_N_WALKERS,
            n_steps=MCMC_SCAN_N_STEPS,
            n_burnin=MCMC_SCAN_N_BURNIN,
        )
        df_scan.to_csv(csv_path, index=False)
        log(f"  Zapisano: {csv_path}")

    log("  Generowanie wykresów MCMC_SCAN...")
    df_scan       = pd.read_csv(csv_path)
    n_identified  = (df_scan["mean_kl_div"]   >= 1.0).sum()
    n_confounded  = (df_scan["max_post_corr"] >= 0.7).sum()
    log(f"  Zidentyfikowane (KL ≥ 1.0): {n_identified}/{len(df_scan)}")
    log(f"  Skonfundowane (corr ≥ 0.7): {n_confounded}/{len(df_scan)}")
    plot_mcmc_scan_summary(df_scan, cfg.param_names, output_dir)
    plot_mcmc_scan_kl(df_scan, cfg.param_names, output_dir)
    plot_mcmc_scan_corr_heatmap(df_scan, cfg.param_names, output_dir)
    plot_mcmc_scan_landscapes(df_scan, cfg.param_names, cfg.bounds, output_dir)
    log(f"  MCMC_SCAN/{cfg.name}: gotowe.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    log("=" * 60)
    log(f"Batch MCMC start — {len(JOBS)} zadań")
    log("=" * 60)

    for i, (agent_key, mode) in enumerate(JOBS, 1):
        cfg = AGENT_CONFIGS[agent_key]
        log(f"\n[{i}/{len(JOBS)}] {mode} → model {cfg.name}")
        t0 = time.time()
        try:
            if mode == "MCMC":
                run_mcmc_job(agent_key, cfg)
            elif mode == "MCMC_SCAN":
                run_mcmc_scan_job(agent_key, cfg)
        except Exception as e:
            log(f"  BŁĄD: {e}")
            import traceback
            traceback.print_exc()
        elapsed = time.time() - t0
        log(f"  Czas: {elapsed/60:.1f} min")

    log("\n" + "=" * 60)
    log("Wszystkie zadania zakończone.")
    log("=" * 60)
