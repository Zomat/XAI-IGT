"""
Batch runner dla analizy PSP (Parameter Space Partitioning) dla wszystkich modeli.
Uruchamia sekwencyjnie bez modyfikacji config.py.

Użycie:
    cd /home/mateusz/projects/igt_sims
    source igt_env/bin/activate
    nohup python src/batch_psp.py > logs/batch_psp.log 2>&1 &
"""

import matplotlib
matplotlib.use("Agg")

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from config import AGENT_CONFIGS
from paths import get_output_dir

from analysis.psp import run_psp_analysis, summarize_psp
from analysis.behavioral import SteingroeverStrategy
from plots.psp import plot_psp_results, plot_psp_distribution

N_SAMPLES = 5000
N_TRIALS  = 100
HORIZON   = 0
SEED      = 42

JOBS = [
    "Agent",         # KPVL
    "PVLDeltaAgent", # PVL-Delta
    "ORLAgent",      # ORL
]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_psp_job(agent_key):
    cfg = AGENT_CONFIGS[agent_key]
    output_dir = get_output_dir(cfg.name, "psp")

    broad = SteingroeverStrategy(restricted=False)
    restr = SteingroeverStrategy(restricted=True)

    for strat in (broad, restr):
        csv_path = output_dir / f"psp_{strat.name.lower()}.csv"

        if csv_path.exists():
            log(f"  PSP/{cfg.name}/{strat.name}: CSV już istnieje — wczytuję i rysuję.")
            df = pd.read_csv(csv_path)
        else:
            log(f"  PSP/{cfg.name}/{strat.name}: generuję {N_SAMPLES} punktów LHS...")
            df = run_psp_analysis(N_SAMPLES, cfg, N_TRIALS, HORIZON, strat, seed=SEED)
            df.to_csv(csv_path, index=False)
            log(f"  Zapisano: {csv_path}")

        _, percents = summarize_psp(df, strat.name)
        plot_psp_results(df, strat.name, output_dir)
        plot_psp_distribution(percents, strat.name, output_dir)
        log(f"  Wykresy zapisane.")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    log("=" * 60)
    log(f"Batch PSP start — {len(JOBS)} modele × 2 strategie")
    log("=" * 60)

    for i, agent_key in enumerate(JOBS, 1):
        cfg = AGENT_CONFIGS[agent_key]
        log(f"\n[{i}/{len(JOBS)}] PSP → model {cfg.name}")
        t0 = time.time()
        try:
            run_psp_job(agent_key)
        except Exception as e:
            log(f"  BŁĄD: {e}")
            import traceback
            traceback.print_exc()
        elapsed = time.time() - t0
        log(f"  Czas: {elapsed/60:.1f} min")

    log("\n" + "=" * 60)
    log("Wszystkie zadania PSP zakończone.")
    log("=" * 60)
