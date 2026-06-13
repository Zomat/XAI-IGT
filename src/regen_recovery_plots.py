"""Regeneruje wykresy recovery_thesis dla wszystkich modeli z istniejących CSV."""
import matplotlib
matplotlib.use("Agg")
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from config import AGENT_CONFIGS
from paths import get_output_dir
from plots.recovery import plot_recovery_thesis

for key in ["Agent", "PVLDeltaAgent", "ORLAgent"]:
    cfg = AGENT_CONFIGS[key]
    out = get_output_dir(cfg.name, "recovery")
    csv = out / "recovery_diagnostics.csv"
    if not csv.exists():
        print(f"Brak {csv} — pomijam")
        continue
    df = pd.read_csv(csv)
    # Nazwy parametrów wynikają z nagłówków CSV (mogą różnić się od aktualnego config)
    param_names_csv = [c[5:] for c in df.columns if c.startswith("true_") and not c.startswith("true_$A") or
                       c.startswith("true_$")]
    param_names_csv = [c[5:] for c in df.columns if c.startswith("true_") and "err" not in c]
    bounds = cfg.bounds
    print(f"Regeneruję recovery dla {cfg.name}, param_names={param_names_csv}...")
    plot_recovery_thesis(df, param_names_csv, bounds, out, model_name=cfg.name)
    print(f"  Done: {cfg.name}")

print("Gotowe.")
