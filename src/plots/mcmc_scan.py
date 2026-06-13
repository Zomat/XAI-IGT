from pathlib import Path
from datetime import datetime
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _lowess(ax, x, y, color="red", lw=2):
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        xy = sorted(zip(x, y))
        xs, ys = zip(*xy)
        sm = lowess(ys, xs, frac=0.6)
        ax.plot(sm[:, 0], sm[:, 1], color=color, lw=lw)
    except Exception:
        m = np.polyfit(x, y, 1)
        xr = np.linspace(min(x), max(x), 50)
        ax.plot(xr, np.poly1d(m)(xr), color=color, lw=lw, ls="--")


# ─────────────────────────────────────────────────────────────────────────────
def plot_mcmc_scan_safe_map(df, param_names, output_dir: Path,
                            kl_threshold: float = 1.0, n_grid: int = 12):
    """Hexbin safe-rate map for MCMC scan — mirrors MLE reliability_slices_safe.

    'Safe' = mean_kl_div >= kl_threshold (posterior meaningfully differs from prior).
    Colormap RdYlGn, 0–100 %, identical style to plot_reliability_slice_heatmaps.
    """
    df = df.copy()

    # detect metric: kl_div (nats, threshold=1.0) or contraction (0-1, threshold=0.5)
    if "mean_kl_div" in df.columns:
        metric_col, threshold, metric_label = "mean_kl_div", kl_threshold, f"KL ≥ {kl_threshold} nat"
    elif any(c.startswith("kl_div_") for c in df.columns):
        kl_cols = [c for c in df.columns if c.startswith("kl_div_")]
        df["mean_kl_div"] = df[kl_cols].mean(axis=1)
        metric_col, threshold, metric_label = "mean_kl_div", kl_threshold, f"KL ≥ {kl_threshold} nat"
    else:
        cont_cols = [c for c in df.columns if c.startswith("contraction_")]
        df["mean_kl_div"] = df[cont_cols].mean(axis=1)
        threshold = 0.5
        metric_col, metric_label = "mean_kl_div", "kontrakcja ≥ 0.5"

    df["identified_rate"] = (df[metric_col] >= threshold).astype(float) * 100.0

    # Spearman ρ: which parameter most drives identifiability
    rhos = {}
    for name in param_names:
        col = f"true_{name}"
        if col in df.columns:
            rho, _ = stats.spearmanr(df[col], df[metric_col])
            rhos[name] = rho
    sorted_by_abs = sorted(rhos, key=lambda n: abs(rhos[n]), reverse=True)
    worst_pair = tuple(sorted_by_abs[:2]) if len(sorted_by_abs) >= 2 else ()

    pairs = list(combinations(param_names, 2))
    n_pairs = len(pairs)
    n_cols = min(4, n_pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.8 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    hb_last = None
    for idx, (p1, p2) in enumerate(pairs):
        ax = axes_flat[idx]
        hb = ax.hexbin(
            df[f"true_{p1}"], df[f"true_{p2}"],
            C=df["identified_rate"],
            gridsize=n_grid,
            cmap="RdYlGn",
            reduce_C_function=np.mean,
            vmin=0, vmax=100,
        )
        hb_last = hb
        ax.set_xlabel(p1, fontsize=11)
        ax.set_ylabel(p2, fontsize=11)
        ax.set_title(f"{p1}\nvs  {p2}", fontsize=11)
        ax.tick_params(labelsize=9)

        rho1 = rhos.get(p1, 0)
        rho2 = rhos.get(p2, 0)
        ax.text(0.02, 0.97,
                f"ρ({p1})={rho1:+.2f}\nρ({p2})={rho2:+.2f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))


    for k in range(n_pairs, len(axes_flat)):
        axes_flat[k].axis("off")

    if hb_last is not None:
        fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.35)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
        fig.colorbar(hb_last, cax=cbar_ax,
                     label=f"Odsetek zidentyfikowanych (%, {metric_label})")

    pct_global = df["identified_rate"].mean()

    fname = output_dir / f"mcmc_scan_safe_map_{_timestamp()}.png"
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_mcmc_scan_kl(df, param_names, output_dir: Path):
    """Per-parameter KL(posterior||prior) as a function of each true parameter value.

    Row = which parameter's KL we're looking at.
    Col = which true parameter dimension drives it.
    Red LOWESS shows the trend.
    """
    KL_THRESHOLD = 1.0

    n = len(param_names)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 3.5 * n), squeeze=False)

    for row, tgt in enumerate(param_names):
        y = df[f"kl_div_{tgt}"].values
        for col, src in enumerate(param_names):
            ax = axes[row, col]
            x = df[f"true_{src}"].values
            ax.scatter(x, y, alpha=0.6, s=40, color="#3498db")
            _lowess(ax, x, y)
            ax.axhline(KL_THRESHOLD, color="gray", ls="--", lw=1, alpha=0.6)
            ax.set_ylim(bottom=-0.05)
            if col == 0:
                ax.set_ylabel(f"KL [naty]\n{tgt}", fontsize=9)
            if row == n - 1:
                ax.set_xlabel(src, fontsize=9)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    fname = output_dir / f"mcmc_scan_kl_{_timestamp()}.png"
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_mcmc_scan_corr_heatmap(df, param_names, output_dir: Path):
    """Two heatmaps: mean and max |posterior correlation| across all scan points.

    High mean |r| for a pair → systematically confounded across parameter space.
    High max |r|             → confounded in at least some regions.
    """
    pairs = list(combinations(param_names, 2))
    n = len(param_names)

    mean_mat = np.zeros((n, n))
    max_mat  = np.zeros((n, n))

    for i, j in combinations(range(n), 2):
        col = f"post_corr_{param_names[i]}_{param_names[j]}"
        if col not in df.columns:
            col = f"post_corr_{param_names[j]}_{param_names[i]}"
        vals = df[col].abs().values
        mean_mat[i, j] = mean_mat[j, i] = float(np.mean(vals))
        max_mat[i, j]  = max_mat[j, i]  = float(np.max(vals))

    np.fill_diagonal(mean_mat, 1.0)
    np.fill_diagonal(max_mat,  1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, mat, title in [
        (ax1, mean_mat, "Średnie |r| posterioru po skanie\n(systematyczne splątanie)"),
        (ax2, max_mat,  "Maksymalne |r| posterioru po skanie\n(najgorszy przypadek splątania)"),
    ]:
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=0, vmax=1, square=True,
                    xticklabels=param_names, yticklabels=param_names,
                    linewidths=0.5, ax=ax)
        ax.set_title(title, fontsize=12)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    fname = output_dir / f"mcmc_scan_corr_heatmap_{_timestamp()}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_mcmc_scan_summary(df, param_names, output_dir: Path):
    """Two-panel summary bar chart per parameter:
      Left  — median KL(posterior||prior) in nats (identifiability;
              median instead of mean: KL distributions are right-skewed)
      Right — mean ESS (sampling quality)
    """
    KL_THRESHOLD = 1.0

    med_kl    = [df[f"kl_div_{n}"].median() for n in param_names]
    means_ess = [df[f"ess_{n}"].mean()      for n in param_names]

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, max(4, len(param_names) * 0.8 + 2)))
    y = np.arange(len(param_names))

    # KL divergence
    colors = ["#2ecc71" if kl >= KL_THRESHOLD else "#e74c3c" for kl in med_kl]
    bars = ax1.barh(y, med_kl, color=colors, edgecolor="white")
    ax1.axvline(KL_THRESHOLD, color="gray", ls="--", lw=1.2)
    ax1.set_yticks(y); ax1.set_yticklabels(param_names)
    ax1.set_xlabel("Mediana KL(a posteriori || a priori)  [naty]", fontsize=10)
    ax1.set_title(f"Identyfikowalność\nzielony ≥ {KL_THRESHOLD} nat = zidentyfikowany", fontsize=11)
    for bar, v in zip(bars, med_kl):
        ax1.text(bar.get_width() + ax1.get_xlim()[1] * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", fontsize=9)

    # ESS
    ess_colors = ["#2ecc71" if e >= 100 else "#e74c3c" for e in means_ess]
    bars3 = ax3.barh(y, means_ess, color=ess_colors, edgecolor="white")
    ax3.axvline(100, color="gray", ls="--", lw=1.2, label="próg ESS=100")
    ax3.set_yticks(y); ax3.set_yticklabels([])
    ax3.set_xlabel("Średnie ESS", fontsize=10)
    ax3.set_title("Efektywna liczba próbek\nzielony ≥ 100 = dobre mieszanie", fontsize=11)
    ax3.legend(fontsize=8)
    for bar, v in zip(bars3, means_ess):
        ax3.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                 f"{v:.0f}", va="center", fontsize=9)

    plt.tight_layout()
    fname = output_dir / f"mcmc_scan_summary_{_timestamp()}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_mcmc_scan_landscapes(df, param_names, bounds, output_dir: Path):
    """Hexbin landscapes for all parameter pairs coloured by max_post_corr.

    Mirrors reliability_slice_heatmaps but uses posterior confounding instead
    of RMSSE — shows WHERE in parameter space the model confounds parameters.
    """
    pairs = list(combinations(param_names, 2))
    n_pairs = len(pairs)
    n_cols = min(5, n_pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4.2 * n_rows),
                              squeeze=False)
    axes_flat = axes.flatten()

    vmax = min(1.0, df["max_post_corr"].quantile(0.95))
    hb_last = None

    for idx, (p1, p2) in enumerate(pairs):
        ax = axes_flat[idx]
        hb = ax.hexbin(
            df[f"true_{p1}"], df[f"true_{p2}"],
            C=df["max_post_corr"],
            gridsize=max(5, len(df) // 4),
            cmap="YlOrRd",
            reduce_C_function=np.mean,
            vmin=0, vmax=vmax,
        )
        hb_last = hb
        ax.set_xlabel(p1, fontsize=9)
        ax.set_ylabel(p2, fontsize=9)
        ax.set_title(f"{p1}\nvs {p2}", fontsize=9)
        ax.tick_params(labelsize=7)

        mean_corr = df["max_post_corr"].mean()
        ax.text(0.02, 0.97, f"mean={mean_corr:.2f}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    for k in range(n_pairs, len(axes_flat)):
        axes_flat[k].axis("off")

    if hb_last is not None:
        fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.35)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
        fig.colorbar(hb_last, cax=cbar_ax, label="Średnie max |r| posterioru")

    fname = output_dir / f"mcmc_scan_landscapes_{_timestamp()}.png"
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
