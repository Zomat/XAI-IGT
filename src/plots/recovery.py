from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_recovery_results(df, horizon, n_trials, param_names, bounds, output_dir: Path):
    """4×2 panel: parameter scatter plots, RMSSE distribution, correlation matrix, summary."""
    true_arr = np.column_stack([df[f"true_{n}"] for n in param_names])
    rec_arr  = np.column_stack([df[f"est_{n}"]  for n in param_names])
    ranges = np.array([b[1] - b[0] for b in bounds])
    scaled_diffs = (true_arr - rec_arr) / ranges
    global_errors = np.sqrt(np.mean(scaled_diffs ** 2, axis=1))

    fig, axes = plt.subplots(4, 2, figsize=(14, 22))
    axes = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes[i]
        x, y = true_arr[:, i], rec_arr[:, i]
        low, high = bounds[i]

        hits = np.sum(np.isclose(y, low, atol=1e-2)) + np.sum(np.isclose(y, high, atol=1e-2))
        hit_pct = (hits / len(y)) * 100

        ax.scatter(x, y, alpha=0.5, edgecolors="none", color="#1f77b4", s=20)
        sns.kdeplot(x=x, y=y, ax=ax, cmap="Blues_d", fill=False, alpha=0.9, linewidths=1.5, levels=6, zorder=2)
        ax.plot([low, high], [low, high], "k--", alpha=0.7, label="Ideal")

        mae = np.mean(np.abs(y - x))
        corr = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{name}\n$r={corr:.2f}$ | $MAE={mae:.2f}$ | Bounds: {hit_pct:.1f}%", fontsize=12)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.grid(True, alpha=0.2)

    # Global error distribution
    ax_err = axes[5]
    sns.histplot(global_errors, bins=25, kde=True, color="orange", ax=ax_err, alpha=0.4, element="step")
    med = np.median(global_errors)
    p90 = np.percentile(global_errors, 90)
    ax_err.axvline(med, color="k", linestyle="-", label=f"Med={med:.3f}")
    ax_err.axvline(p90, color="k", linestyle="--", label=f"P90={p90:.3f}")
    ax_err.set_title("Global Estimation Error (RMSSE)", fontsize=14, fontweight="bold")
    ax_err.set_xlabel("RMSSE (Scaled)")
    ax_err.legend()

    # Correlation matrix of recovered params
    ax_corr = axes[6]
    corr_matrix = np.corrcoef(rec_arr.T)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                ax=ax_corr, cbar=False)
    ax_corr.set_title("Recovered Params Correlation", fontsize=12)
    ax_corr.set_xticklabels(param_names, rotation=45, ha="right")
    ax_corr.set_yticklabels(param_names, rotation=0)

    # Summary text
    ax_info = axes[7]
    ax_info.axis("off")
    ax_info.text(
        0.1, 0.5,
        f"RECOVERY SUMMARY\n{'─'*20}\n"
        f"Mean RMSSE:      {np.mean(global_errors):.3f}\n"
        f"Median RMSSE:    {med:.3f}\n"
        f"90th Percentile: {p90:.3f}\n"
        f"Total Samples:   {len(true_arr)}\n"
        f"Failed Fits:     {np.sum(np.isnan(rec_arr[:, 0]))}",
        fontsize=14, family="monospace", verticalalignment="center",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = output_dir / f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_recovery_thesis(df, param_names, bounds, output_dir: Path, model_name: str = ""):
    """Wykresy odtwarzania parametrów do pracy magisterskiej.

    Sama siatka rozrzutu (prawdziwa vs odtworzona wartość) — bez RMSSE i macierzy korelacji.
    Układ: max 3 kolumny, liczba wierszy dopasowana do liczby parametrów.
    """
    true_arr = np.column_stack([df[f"true_{n}"] for n in param_names])
    rec_arr  = np.column_stack([df[f"est_{n}"]  for n in param_names])
    n_params = len(param_names)

    n_cols = min(n_params, 3)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 5.8 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes_flat[i]
        x, y = true_arr[:, i], rec_arr[:, i]
        low, high = bounds[i]

        pad = (high - low) * 0.05
        ax.scatter(x, y, alpha=0.35, edgecolors="none", color="#2980b9", s=18, zorder=2)
        try:
            sns.kdeplot(x=x, y=y, ax=ax, cmap="Blues_d", fill=False,
                        alpha=0.85, linewidths=1.2, levels=5, zorder=3)
        except Exception:
            pass
        ax.plot([low, high], [low, high], color="black", ls="--", lw=1.5,
                alpha=0.6, zorder=4, label="Idealne odtwarzanie")

        hits     = np.sum(np.isclose(y, low, atol=1e-2)) + np.sum(np.isclose(y, high, atol=1e-2))
        hit_pct  = hits / len(y) * 100
        corr = np.corrcoef(x, y)[0, 1]
        mae  = np.mean(np.abs(y - x)) / (high - low)
        ax.set_title(f"{name}\n$r = {corr:.2f}$,  $MAE = {mae:.3f}$,  granice: ${hit_pct:.1f}\\%$", fontsize=14)
        ax.set_xlabel("Wartość rzeczywista", fontsize=13)
        ax.set_ylabel("Wartość odtworzona", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.grid(True, alpha=0.2, linewidth=0.7)
        ax.set_aspect("equal", adjustable="box")

    for k in range(n_params, len(axes_flat)):
        axes_flat[k].set_visible(False)

    title = f"Odtwarzanie parametrów — {model_name}  ($N = {len(true_arr)}$)" if model_name else \
            f"Odtwarzanie parametrów modelu  ($N = {len(true_arr)}$)"
    plt.tight_layout(h_pad=4.0)
    fname = output_dir / f"recovery_thesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
