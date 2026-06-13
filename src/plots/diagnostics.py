from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_full_identification_matrix(npz_path, param_names, output_dir: Path,
                                    model_name: str = ""):
    """Profil NLL (górny rząd) + krajobrazy 2D NLL (dolne rzędy)."""
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec

    data = np.load(npz_path)
    true_params = data["true_params"]
    mle_params  = data["mle_params"]
    profile_x   = data["profile_x"]
    profile_nll = data["profile_nll"]
    grid_axes_x = data["grid_axes_x"]
    grid_axes_y = data["grid_axes_y"]
    grid_Z      = data["grid_Z"]

    n_params = len(param_names)
    pairs = [(i, j) for i in range(n_params) for j in range(i)]
    n_pairs = len(pairs)
    n_landscape_rows = int(np.ceil(n_pairs / n_params))
    n_content_rows   = 1 + n_landscape_rows  # profiles + landscapes

    # GridSpec: content rows + 1 dedicated legend/colorbar row at the bottom
    fig = plt.figure(figsize=(5 * n_params, 5 * n_content_rows + 2.5))
    gs = GridSpec(
        n_content_rows + 1, n_params,
        figure=fig,
        height_ratios=[1.0] + [1.1] * n_landscape_rows + [0.38],
        hspace=0.55,
        wspace=0.38,
        top=0.92,    # leaves ~8 % gap below suptitle
        bottom=0.02,
        left=0.06,
        right=0.98,
    )

    title = f"Macierz identyfikowalności parametrów modelu {model_name}" if model_name \
            else "Macierz identyfikowalności parametrów"

    # ── Row 0: 1D NLL profiles ────────────────────────────────────────────
    for i in range(n_params):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(profile_x[:, i], profile_nll[:, i], color="teal", lw=3)
        ax.axvline(true_params[i], color="black",   ls="--", lw=2.2)
        ax.axvline(mle_params[i],  color="#e74c3c", ls=":",  lw=2.8)
        ax.set_title(param_names[i], fontsize=17, fontweight="bold", pad=8)
        ax.set_ylabel("NLL", fontsize=14, color="gray")
        ax.set_xlabel("wartość parametru", fontsize=12, color="gray")
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.25, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Rows 1+: 2D NLL landscapes ───────────────────────────────────────
    for idx, (i, j) in enumerate(pairs):
        row = 1 + idx // n_params
        col = idx % n_params
        ax  = fig.add_subplot(gs[row, col])
        Z   = grid_Z[i, j]
        levels = np.linspace(Z.min(), Z.min() + (Z.max() - Z.min()) * 0.2, 15)
        ax.contourf(grid_axes_x[i, j], grid_axes_y[i, j], Z,
                    levels=levels, cmap="viridis_r", alpha=0.85)
        ax.plot(true_params[j], true_params[i], "o", color="white",
                ms=11, mec="black", mew=2, zorder=5)
        ax.plot(mle_params[j],  mle_params[i],  "s", color="#e74c3c",
                ms=10, mec="white", mew=1.2, zorder=6)
        ax.set_xlabel(param_names[j], fontsize=13, fontweight="bold")
        ax.set_ylabel(param_names[i], fontsize=13, fontweight="bold")
        ax.tick_params(labelsize=11)

    # Hide unused cells in the last landscape row
    for idx in range(n_pairs, n_landscape_rows * n_params):
        fig.add_subplot(gs[1 + idx // n_params, idx % n_params]).axis("off")

    # ── Bottom row: legend (left) + colorbar (right) — dedicated space ───
    bottom_ax = fig.add_subplot(gs[n_content_rows, :])
    bottom_ax.axis("off")
    pos = bottom_ax.get_position()   # figure coordinates of this row

    legend_elements = [
        Line2D([0], [0], color="teal",    lw=5,    label="Profil NLL"),
        Line2D([0], [0], color="black",   ls="--", lw=4,   label="Prawdziwa wartość parametru"),
        Line2D([0], [0], color="#e74c3c", ls=":",  lw=4.5, label="Estymata MLE"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=2.5, ms=18,
               label="Prawdziwa wartość (wykres 2D)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e74c3c",
               markeredgecolor="white", ms=17, label="Estymata MLE (wykres 2D)"),
    ]
    bottom_ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        fontsize=15,
        title="Legenda",
        title_fontsize=17,
        framealpha=0.97,
        edgecolor="#aaaaaa",
        borderpad=1.1,
        labelspacing=0.7,
        handlelength=3.5,
        handletextpad=1.0,
        ncols=2,
    )

    # Colorbar in the right half of the bottom row
    cbar_ax = fig.add_axes([
        pos.x0 + 0.56 * pos.width,
        pos.y0 + 0.22 * pos.height,
        0.42 * pos.width,
        0.30 * pos.height,
    ])
    sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_ticks([0.02, 0.98])
    cb.set_ticklabels(["niskie NLL  (dobry obszar)", "wysokie NLL  (zły obszar)"],
                      fontsize=14)
    cb.ax.set_title("Kolor na wykresach 2D:", fontsize=15, pad=8, loc="left")
    cb.outline.set_linewidth(1.2)

    fname = output_dir / f"identification_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
    print(f"Saved: {fname}")


def plot_full_diagnostic_matrix(df: pd.DataFrame, true_params, param_names, bounds, output_dir: Path):
    """Pairwise MLE cloud matrix and signed error coupling matrix."""
    true_p = np.array(true_params)
    n_params = len(param_names)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    est_params = df[param_names].values
    df_est = df[param_names].copy()
    ranges = np.array([b[1] - b[0] for b in bounds])
    scaled_errors = (est_params - true_p) / ranges
    df_err = pd.DataFrame(scaled_errors, columns=[f"Δ{n}" for n in param_names])

    # MLE cloud matrix
    g = sns.PairGrid(df_est, corner=True, diag_sharey=False)
    g.map_lower(sns.scatterplot, alpha=0.25, s=12, color="#3498db", linewidths=0)
    g.map_diag(sns.kdeplot, color="#2c3e50", fill=True, alpha=0.15, linewidth=1.5)
    # ── Dekoracje (linie + punkt prawdziwy) ──────────────────────────────
    for i in range(n_params):
        for j in range(i):
            ax = g.axes[i, j]
            ax.axvline(true_p[j], color="#e74c3c", ls="--", alpha=0.7, lw=1.2)
            ax.axhline(true_p[i], color="#e74c3c", ls="--", alpha=0.7, lw=1.2)
            ax.scatter(true_p[j], true_p[i], color="#e74c3c", marker="x",
                       s=60, lw=1.8, zorder=10)
            ax.spines[["top", "right"]].set_visible(False)
        g.axes[i, i].axvline(true_p[i], color="#e74c3c", ls="--", alpha=0.7, lw=1.2)
        g.axes[i, i].spines[["top", "right"]].set_visible(False)

    # Seaborn suppresses y-axis labels on diagonal axes; for the top-left plot
    # there is no scatter to its left, so we add the label via text annotation.
    g.axes[0, 0].text(
        -0.12, 0.5, param_names[0],
        transform=g.axes[0, 0].transAxes,
        fontsize=11,
        ha="right", va="center", rotation=90,
    )
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#e74c3c", ls="--", lw=1.8,
               label="Prawdziwa wartość parametru"),
        Line2D([0], [0], marker="x", color="#e74c3c", lw=0,
               markersize=8, markeredgewidth=2, label="Punkt prawdziwy"),
    ]
    g.figure.legend(handles=legend_elems, fontsize=10,
                    framealpha=0.95, edgecolor="#cccccc",
                    loc="upper right",
                    bbox_to_anchor=(0.98, 0.97))

    # ── Stałe zakresy osi — ustawiane jako ostatnie, po wszystkich dekoracjach ──
    # Iterujemy po każdej osi osobno (nie polegamy na propagacji przez sharex/sharey)
    for i in range(n_params):
        for j in range(i + 1):   # dolny trójkąt + przekątna
            ax = g.axes[i, j]
            if ax is None:
                continue
            ax.set_xlim(bounds[j][0], bounds[j][1])
            if i != j:           # poza przekątną: również oś Y
                ax.set_ylim(bounds[i][0], bounds[i][1])

    fname = output_dir / f"diagnostic_mle_matrix_{timestamp}.png"
    g.figure.savefig(fname, dpi=220, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved: {fname}")

    # Error coupling matrix
    df_err.columns = [f"Δ{n}" for n in param_names]
    # Symetryczny zakres osi per parametr (percentyl 99 błędów)
    err_lims = [float(np.percentile(np.abs(scaled_errors[:, k]), 99)) * 1.1
                for k in range(n_params)]

    h = sns.PairGrid(df_err, corner=True, diag_sharey=False)
    h.map_lower(sns.scatterplot, alpha=0.25, s=12, color="#e67e22", linewidths=0)
    h.map_diag(sns.kdeplot, color="#e67e22", fill=True, alpha=0.15, linewidth=1.5)
    for i in range(n_params):
        for j in range(n_params):
            ax = h.axes[i, j]
            if ax is None:
                continue
            lim_x = err_lims[j]
            lim_y = err_lims[i]
            ax.set_xlim(-lim_x, lim_x)
            if i != j:
                ax.set_ylim(-lim_y, lim_y)
            ax.axvline(0, color="black", ls="--", alpha=0.35, lw=1)
            ax.axhline(0, color="black", ls="--", alpha=0.35, lw=1)
            ax.spines[["top", "right"]].set_visible(False)
    fname = output_dir / f"diagnostic_error_coupling_{timestamp}.png"
    h.figure.savefig(fname, dpi=220, bbox_inches="tight")
    plt.close(h.figure)
    print(f"Saved: {fname}")


def plot_parameter_error_profiles(df: pd.DataFrame, true_params, param_names, bounds, output_dir: Path):
    """Histogram of absolute scaled errors per parameter + global RMSSE."""
    true_p = np.array(true_params)
    est_params = df[param_names].values
    ranges = np.array([b[1] - b[0] for b in bounds])
    abs_scaled = np.abs(est_params - true_p) / ranges

    n_params = len(param_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(n_params):
        ax = axes[i]
        sns.histplot(abs_scaled[:, i], kde=True, ax=ax, color="gray", alpha=0.5)
        lines = ax.get_lines()
        if lines:
            lines[0].set_color("darkorange")
        med = np.median(abs_scaled[:, i])
        p90 = np.percentile(abs_scaled[:, i], 90)
        ax.axvline(med, color="k", label=f"med={med:.3f}")
        ax.axvline(p90, color="k", ls="--", label=f"p90={p90:.3f}")
        ax.set_title(f"Error Profile: {param_names[i]}")
        ax.set_xlim(0, 1)
        ax.legend()

    # Slot 5 (index 5): global RMSSE
    global_errors = np.sqrt(np.mean(abs_scaled ** 2, axis=1))
    sns.histplot(global_errors, kde=True, ax=axes[5], color="blue", alpha=0.2)
    axes[5].set_title("GLOBAL ERROR (RMSSE)")
    axes[5].set_xlim(0, 1)

    # Hide unused axes when n_params < 5
    for k in range(n_params, 5):
        axes[k].axis("off")

    plt.tight_layout()
    fname = output_dir / f"error_profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"Saved: {fname}")
