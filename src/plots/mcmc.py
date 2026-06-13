from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def plot_mcmc_traces(chain, param_names, output_dir: Path):
    """Trace plot: sample value vs step for every walker and parameter.

    A well-mixed chain looks like a 'hairy caterpillar'.
    Drifting or stratified traces indicate poor mixing / identifiability issues.
    """
    n_steps, n_walkers, n_params = chain.shape

    fig, axes = plt.subplots(n_params, 1, figsize=(14, 3 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(chain[:, :, i], color="steelblue", alpha=0.15, lw=0.6)
        ax.plot(chain[:, :, i].mean(axis=1), color="black", lw=1.5, label="średnia")
        ax.set_ylabel(name, fontsize=10)
        ax.yaxis.set_label_position("right")

    axes[-1].set_xlabel("Krok")
    plt.tight_layout()
    fname = output_dir / f"mcmc_traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_mcmc_corner(flat_samples, param_names, bounds, true_params, output_dir: Path):
    """Corner plot: pairwise posterior distributions with marginals on the diagonal.

    Elongated off-diagonal clouds → parameter degeneracy (one combination of
    params is constrained, but not each individually).
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_params, figsize=(4 * n_params, 4 * n_params))
    true_p = np.array(true_params)

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]

            if row == col:
                # Diagonal: marginal posterior
                sns.kdeplot(flat_samples[:, row], ax=ax, fill=True,
                            color="steelblue", alpha=0.4, linewidth=1.5)
                ax.axvline(true_p[row], color="red",   ls="--", lw=1.5, label="prawdziwa")
                ax.axvline(flat_samples[:, row].mean(), color="black", ls="-",  lw=1, label="średnia")
                lo, hi = bounds[row]
                ax.set_xlim(lo, hi)
                ax.set_yticks([])
                if row == 0:
                    ax.legend(fontsize=11)

            elif row > col:
                # Lower triangle: joint posterior
                ax.scatter(flat_samples[::10, col], flat_samples[::10, row],
                           alpha=0.15, s=4, color="steelblue", rasterized=True)
                try:
                    sns.kdeplot(x=flat_samples[:, col], y=flat_samples[:, row],
                                ax=ax, levels=4, color="navy", linewidths=0.8)
                except Exception:
                    pass
                ax.scatter(true_p[col], true_p[row], color="red",
                           s=60, marker="x", zorder=10, linewidths=2)
                ax.set_xlim(*bounds[col])
                ax.set_ylim(*bounds[row])

            else:
                # Upper triangle: correlation coefficient
                corr = np.corrcoef(flat_samples[:, col], flat_samples[:, row])[0, 1]
                ax.text(0.5, 0.5, f"r = {corr:.2f}",
                        ha="center", va="center", fontsize=18,
                        color="darkred" if abs(corr) > 0.5 else "black",
                        fontweight="bold" if abs(corr) > 0.5 else "normal",
                        transform=ax.transAxes)
                ax.set_facecolor("#f7f7f7")
                ax.set_xticks([])
                ax.set_yticks([])

            # Axis labels on edges only
            if col == 0 and row > 0:
                ax.set_ylabel(param_names[row], fontsize=13)
            else:
                ax.set_ylabel("")
            if row == n_params - 1 and col < n_params - 1:
                ax.set_xlabel(param_names[col], fontsize=13)
            else:
                ax.set_xlabel("")

            ax.tick_params(labelsize=10)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    fig.subplots_adjust(top=0.95)
    fname = output_dir / f"mcmc_corner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_mcmc_summary(df_diags: "pd.DataFrame", param_names, bounds, output_dir: Path):
    """Two-panel summary:
      Left  — KL(posterior || prior) per parameter in nats
      Right — 95% credible intervals vs prior range

    df_diags: DataFrame loaded from mcmc_diags.csv with columns
      param, kl_div, mean, ci_lo, ci_hi (one row per parameter).
    """
    import pandas as pd
    KL_THRESHOLD = 1.0   # nats; below = data barely informative beyond prior

    d = df_diags.set_index("param")
    n_params = len(param_names)
    kl_vals = [d.loc[n, "kl_div"] for n in param_names]
    means   = [d.loc[n, "mean"]   for n in param_names]
    ci_lo   = [d.loc[n, "ci_lo"]  for n in param_names]
    ci_hi   = [d.loc[n, "ci_hi"]  for n in param_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, n_params * 0.9 + 2)))
    y_pos = np.arange(n_params)
    colors = ["#2ecc71" if kl >= KL_THRESHOLD else "#e74c3c" for kl in kl_vals]

    # --- Left: KL divergence bar chart ---
    bars = ax1.barh(y_pos, kl_vals, color=colors, edgecolor="white", height=0.6)
    ax1.axvline(KL_THRESHOLD, color="gray", ls="--", lw=1.2,
                label=f"threshold ({KL_THRESHOLD} nat)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(param_names, fontsize=10)
    ax1.set_xlabel("KL(posterior || prior)  [naty]", fontsize=11)
    ax1.set_title("Identyfikowalność parametrów\n"
                  f"zielony ≥ {KL_THRESHOLD} nat = zidentyfikowany   czerwony = niezidentyfikowany",
                  fontsize=10)
    ax1.legend(fontsize=9)
    for bar, kl in zip(bars, kl_vals):
        ax1.text(bar.get_width() + ax1.get_xlim()[1] * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{kl:.2f}", va="center", fontsize=9)

    # --- Right: credible interval vs prior range ---
    for i, name in enumerate(param_names):
        lo_bound, hi_bound = bounds[i]
        ax2.barh(i, hi_bound - lo_bound, left=lo_bound,
                 color="lightgray", height=0.4, label="Prior" if i == 0 else "")
        ax2.barh(i, ci_hi[i] - ci_lo[i], left=ci_lo[i],
                 color=colors[i], height=0.4, alpha=0.85,
                 label="95% PI posterioru" if i == 0 else "")
        ax2.scatter(means[i], i, color="black", s=40, zorder=5)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(param_names, fontsize=10)
    ax2.set_xlabel("Wartość parametru", fontsize=11)
    ax2.set_title("95% PI posterioru vs zakres prioru\n(czarny punkt = średnia posterioru)", fontsize=11)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fname = output_dir / f"mcmc_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
