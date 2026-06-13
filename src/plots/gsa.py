from pathlib import Path
from datetime import datetime
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_gsa_marginal(df, param_names, output_dir: Path):
    """Marginal reliability: global error as a function of each true parameter."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(1, len(param_names), figsize=(5 * len(param_names), 6), sharey=True)
    if len(param_names) == 1:
        axes = [axes]

    for i, name in enumerate(param_names):
        sns.regplot(
            data=df, x=f"true_{name}", y="global_error",
            lowess=True, ax=axes[i],
            scatter_kws={"alpha": 0.15, "color": "gray"},
            line_kws={"color": "red", "lw": 3},
        )
        axes[i].set_title(f"Reliability: {name}")
        axes[i].set_ylim(0, 0.6)

    plt.tight_layout()
    fname = output_dir / f"gsa_marginal_{timestamp}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_gsa_coupling(df, param_names, output_dir: Path):
    """Heatmap of error correlations across parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_cols = [f"err_{name}" for name in param_names]
    corr_matrix = df[error_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        xticklabels=param_names, yticklabels=param_names,
        square=True, cbar_kws={"shrink": 0.8}, ax=ax,
    )

    fname = output_dir / f"gsa_coupling_{timestamp}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_all_error_landscapes(df, param_names, output_dir: Path):
    """Hexbin error landscape for every pair of parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pairs = list(itertools.combinations(param_names, 2))
    n_pairs = len(pairs)

    n_cols = 5
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    vmax = df["global_error"].quantile(0.95)
    vmin = df["global_error"].min()

    hb = None
    for i, (p1, p2) in enumerate(pairs):
        ax = axes[i]
        hb = ax.hexbin(
            df[f"true_{p1}"], df[f"true_{p2}"],
            C=df["global_error"],
            gridsize=15, cmap="YlOrRd",
            reduce_C_function=np.mean,
            vmin=vmin, vmax=vmax,
        )
        ax.set_xlabel(f"True {p1}", fontsize=10)
        ax.set_ylabel(f"True {p2}", fontsize=10)
        ax.set_title(f"{p1} vs {p2}", fontsize=11, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if hb is not None:
        fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.35)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(hb, cax=cbar_ax, label="Mean Global Error (RMSSE)")

    fname = output_dir / f"gsa_landscapes_{timestamp}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_error_normality(df, param_names, output_dir: Path):
    """Violin plots and Shapiro-Wilk test results for error distributions."""
    n_params = len(param_names)
    sample_agents = df["agent_id"].unique()[:10]
    df_sample = df[df["agent_id"].isin(sample_agents)]

    # Violin plots
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for i, name in enumerate(param_names):
        sns.violinplot(
            data=df_sample, x="agent_id", y=f"err_{name}",
            ax=axes[i], inner="quartile", palette="muted",
        )
        axes[i].axhline(0, color="red", linestyle="--", alpha=0.6)
        axes[i].set_title(f"Error distribution: {name} (10 agents)")
        axes[i].set_ylabel("Estimation Error")

    plt.xlabel("Agent ID")
    plt.tight_layout()
    fname = output_dir / "normality_violins.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"Saved: {fname}")

    # Shapiro-Wilk p-value histograms
    shapiro_results = {name: [] for name in param_names}
    for _, group in df.groupby("agent_id"):
        for name in param_names:
            errors = group[f"err_{name}"].values
            if len(errors) >= 3:
                _, p_val = stats.shapiro(errors)
                shapiro_results[name].append(p_val)

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    for i, name in enumerate(param_names):
        p_vals = shapiro_results[name]
        sns.histplot(p_vals, bins=20, ax=axes[i], color="skyblue")
        axes[i].axvline(0.05, color="red", linestyle="--", label="α=0.05")
        pct_normal = (np.array(p_vals) > 0.05).mean() * 100
        axes[i].set_title(f"{name}\nNormal: {pct_normal:.1f}% of agents")
        axes[i].set_xlabel("Shapiro-Wilk p-value")
        if i == 0:
            axes[i].legend()

    plt.tight_layout()
    fname = output_dir / "normality_shapiro.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_reliability_map(df, param_names, output_dir: Path):
    """Visualise where in parameter space the model is reliable.

    Produces three figures:
      1. reliability_rmsse.png  — RMSSE (median & p90) vs each true parameter
      2. reliability_normality.png — Shapiro-Wilk p-value per parameter vs each true parameter
      3. reliability_bias.png   — signed bias per parameter vs each true parameter

    Args:
        df: output of run_reliability_scan()
        param_names: list of parameter names
        output_dir: directory to save PNGs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_params = len(param_names)

    # ── 1. RMSSE map ─────────────────────────────────────────────────────────
    fig_w = max(13, 2.8 * n_params + 1)
    fig, axes = plt.subplots(2, n_params, figsize=(fig_w, 7), sharey="row")

    for col, name in enumerate(param_names):
        x_col = f"true_{name}"

        # Row 0: median RMSSE
        ax = axes[0, col]
        ax.scatter(df[x_col], df["rmsse_median"], alpha=0.5, s=25, color="#2980b9")
        _add_lowess(ax, df[x_col], df["rmsse_median"], color="red")
        ax.set_title(name, fontsize=16, fontweight="bold")
        ax.set_ylabel("Mediana RMSSE" if col == 0 else "", fontsize=14)
        ax.set_ylim(bottom=0)
        ax.axhline(0.2, color="gray", ls="--", lw=1.2, alpha=0.6, label="próg 0,2")
        ax.grid(True, alpha=0.2, lw=0.7)
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.legend(fontsize=12)

        # Row 1: p90 RMSSE
        ax2 = axes[1, col]
        ax2.scatter(df[x_col], df["rmsse_p90"], alpha=0.5, s=25, color="#e67e22")
        _add_lowess(ax2, df[x_col], df["rmsse_p90"], color="red")
        ax2.set_ylabel("RMSSE (percentyl 90.)" if col == 0 else "", fontsize=14)
        ax2.set_xlabel(name, fontsize=14)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.2, lw=0.7)
        ax2.tick_params(labelsize=12)
        ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = output_dir / f"reliability_rmsse_{timestamp}.png"
    fig.savefig(fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

    # ── 2. Normality map ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_params, n_params, figsize=(4 * n_params, 4 * n_params))

    sc = None
    for row, err_name in enumerate(param_names):
        p_col = f"shapiro_p_{err_name}"
        for col, x_name in enumerate(param_names):
            ax = axes[row, col]
            sc = ax.scatter(df[f"true_{x_name}"], df[p_col],
                            c=df[p_col], cmap="RdYlGn",
                            vmin=0, vmax=0.2, alpha=0.7, s=35)
            ax.axhline(0.05, color="red", ls="--", lw=1.2, alpha=0.8)
            if col == 0:
                ax.set_ylabel(f"SW p: {err_name}", fontsize=9)
            if row == n_params - 1:
                ax.set_xlabel(x_name, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="p-wartość (Shapiro-Wilk)")
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    fname = output_dir / f"reliability_normality_{timestamp}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

    # ── 3. Bias map ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_params, n_params, figsize=(4 * n_params, 4 * n_params))

    sc = None
    for row, err_name in enumerate(param_names):
        bias_col = f"bias_{err_name}"
        bias_abs_max = df[bias_col].abs().quantile(0.95) + 1e-6
        for col, x_name in enumerate(param_names):
            ax = axes[row, col]
            sc = ax.scatter(df[f"true_{x_name}"], df[bias_col],
                            c=df[bias_col], cmap="RdBu_r",
                            vmin=-bias_abs_max, vmax=bias_abs_max,
                            alpha=0.7, s=35)
            ax.axhline(0, color="black", ls="-", lw=0.9, alpha=0.4)
            if col == 0:
                ax.set_ylabel(f"Obciążenie: {err_name}", fontsize=9)
            if row == n_params - 1:
                ax.set_xlabel(x_name, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="Obciążenie (est − prawda)")
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    fname = output_dir / f"reliability_bias_{timestamp}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_reliability_parallel_coords(df, param_names, bounds, output_dir: Path,
                                     rmsse_threshold=0.2):
    """Parallel coordinates plot coloured by reliability.

    Each line = one sampled parameter vector.
    Green = Safe  (rmsse_median < threshold)
    Red   = Unsafe

    Best way to see which parameter *combinations* cause poor identifiability:
    red lines tend to cluster at specific positions on certain axes.
    """
    n_params = len(param_names)
    safe_mask = df["rmsse_median"] < rmsse_threshold
    n_safe   = safe_mask.sum()
    n_unsafe = (~safe_mask).sum()

    # Normalise each parameter to [0, 1] using its bounds
    norm_vals = np.zeros((len(df), n_params))
    for j, (name, (lo, hi)) in enumerate(zip(param_names, bounds)):
        norm_vals[:, j] = (df[f"true_{name}"].values - lo) / (hi - lo)

    x_pos = np.arange(n_params)

    fig, ax = plt.subplots(figsize=(max(10, n_params * 2.2), 7))

    # Draw unsafe lines first (underneath), then safe on top
    for mask, color, alpha, zorder in [
        (~safe_mask, "#e74c3c", 0.25, 1),
        ( safe_mask, "#27ae60", 0.40, 2),
    ]:
        for idx in df.index[mask]:
            ax.plot(x_pos, norm_vals[idx], color=color, alpha=alpha, lw=0.9, zorder=zorder)

    # Vertical axes
    for j, (name, (lo, hi)) in enumerate(zip(param_names, bounds)):
        ax.axvline(j, color="black", lw=0.8, alpha=0.6, zorder=3)
        ax.text(j,  1.04, f"{hi:.2g}", ha="center", va="bottom", fontsize=8, color="gray")
        ax.text(j, -0.04, f"{lo:.2g}", ha="center", va="top",    fontsize=8, color="gray")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_names, fontsize=11, fontweight="bold")
    ax.set_ylim(-0.12, 1.12)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["min", "śr", "max"], fontsize=9)
    ax.set_ylabel("Znormalizowana wartość parametru", fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#27ae60", lw=2.5,
               label=f"Dobry — RMSSE < {rmsse_threshold}  (n={n_safe}, {n_safe/len(df)*100:.0f}%)"),
        Line2D([0], [0], color="#e74c3c", lw=2.5,
               label=f"Słaby — RMSSE ≥ {rmsse_threshold}  (n={n_unsafe}, {n_unsafe/len(df)*100:.0f}%)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fname = output_dir / f"reliability_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_reliability_slice_heatmaps(df, param_names, _bounds, output_dir: Path,
                                    rmsse_threshold=0.2, n_grid=12):
    """2D slice heatmaps for ALL parameter pairs — Safe Rate and Mean RMSSE.

    For each pair (p_i, p_j) the remaining dimensions are projected out via
    hexbin averaging, which is equivalent to marginalising over them (valid
    under LHS sampling).

    Produces three files:
      reliability_slices_safe_*.png  — Safe Rate (%) for every pair
      reliability_slices_rmsse_*.png — Mean RMSSE for every pair
      reliability_spearman_*.png     — Spearman |ρ| ranking bar chart
      (worst pair is highlighted with a red border in both heatmap figures)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = df.copy()
    df["safe_rate"] = (df["rmsse_median"] < rmsse_threshold).astype(float) * 100.0

    # ── Spearman ρ per parameter ──────────────────────────────────────────────
    rhos = {}
    for name in param_names:
        rho, _ = stats.spearmanr(df[f"true_{name}"], df["rmsse_median"])
        rhos[name] = rho
    sorted_by_abs = sorted(rhos, key=lambda n: abs(rhos[n]), reverse=True)
    worst_pair = tuple(sorted_by_abs[:2])

    pairs = list(itertools.combinations(param_names, 2))
    n_pairs = len(pairs)

    # Grid layout: up to 4 columns for better readability
    n_cols = min(4, n_pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))

    vmax_rmsse = df["rmsse_median"].quantile(0.95)

    for metric, cmap, label, fname_stem, vmin, vmax in [
        ("safe_rate",    "RdYlGn", f"Odsetek dobrych estymacji (%, RMSSE < {rmsse_threshold})",
         "reliability_slices_safe",   0,    100),
        ("rmsse_median", "YlOrRd_r",  "Średnia RMSSE (mediana)",
         "reliability_slices_rmsse",  0,    vmax_rmsse),
    ]:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.8 * n_cols, 3.8 * n_rows),
                                 squeeze=False)
        axes_flat = axes.flatten()

        hb_last = None
        for idx, (p1, p2) in enumerate(pairs):
            ax = axes_flat[idx]
            hb = ax.hexbin(
                df[f"true_{p1}"], df[f"true_{p2}"],
                C=df[metric],
                gridsize=n_grid,
                cmap=cmap,
                reduce_C_function=np.mean,
                vmin=vmin, vmax=vmax,
            )
            hb_last = hb
            ax.set_xlabel(p1, fontsize=11)
            ax.set_ylabel(p2, fontsize=11)
            ax.tick_params(labelsize=9)

            rho1, rho2 = rhos[p1], rhos[p2]
            ax.set_title(f"{p1}\nvs  {p2}", fontsize=11)

            # Annotate Spearman ρ values in corner
            ax.text(0.02, 0.97,
                    f"ρ({p1})={rho1:+.2f}\nρ({p2})={rho2:+.2f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    color="black", alpha=0.7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

        # Hide unused axes
        for k in range(n_pairs, len(axes_flat)):
            axes_flat[k].axis("off")

        # Shared colorbar
        fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.35)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
        fig.colorbar(hb_last, cax=cbar_ax, label=label)

        fname = output_dir / f"{fname_stem}_{timestamp}.png"
        fig.savefig(fname, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")

    # ── Spearman ρ ranking bar chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(param_names) * 1.5), 4))
    rho_vals = [rhos[n] for n in sorted_by_abs]
    rho_abs  = [abs(v)  for v in rho_vals]
    colors   = ["#e74c3c" if v > 0 else "#3498db" for v in rho_vals]

    bars = ax.bar(sorted_by_abs, rho_abs, color=colors, edgecolor="white")
    ax.axhline(0.3, color="gray", ls="--", lw=1, label="|ρ| = 0,3 (umiarkowany)")
    for bar, rho in zip(bars, rho_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rho:+.2f}", ha="center", fontsize=10)
    ax.set_ylabel("|Spearman ρ|  z  RMSSE", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname3 = output_dir / f"reliability_spearman_{timestamp}.png"
    fig.savefig(fname3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname3}")


def plot_reliability_quadrant(df, output_dir: Path,
                               rmsse_threshold=0.2, normality_threshold=0.05):
    """2×2 scatter: RMSSE vs mediana p-wartości Shapiro-Wilka.

    Klasyfikuje każdy punkt przestrzeni parametrów do jednego z 4 kwadrantów:
      DOBRY            — mały błąd + rozkład normalny  → CI wiarygodne
      Przewidywalnie słaby — duży błąd + normalny      → CI poprawne, ale duże
      NAJGORSZY        — mały błąd + nienormalny        → CI FAŁSZYWE, niewidoczne
      Zły              — duży błąd + nienormalny        → CI FAŁSZYWE, widoczne

    Oś Y: mediana p-wartości Shapiro-Wilka po wszystkich parametrach (ciągła).
    Obliczana z kolumn shapiro_p_* jeśli shapiro_p_median nie istnieje w df.
    """
    from matplotlib.patches import Rectangle

    # ── Wyznacz metrykę Y ─────────────────────────────────────────────────
    if "shapiro_p_median" not in df.columns:
        shapiro_cols = sorted(c for c in df.columns if c.startswith("shapiro_p_"))
        df = df.copy()
        df["shapiro_p_median"] = df[shapiro_cols].median(axis=1)

    x = df["rmsse_median"].values
    y = np.clip(df["shapiro_p_median"].values, 1e-6, 1.0)
    n = len(df)

    xt   = rmsse_threshold
    yt   = normality_threshold                          # 0.05
    xmax = max(x.max() * 1.12, xt * 2.2)
    ymin = max(y.min() * 0.3,  1e-5)
    ymax = max(y.max() * 2.0,  yt  * 20)
    xlim = (0, xmax)
    ylim = (ymin, ymax)


    QUADRANTS = [
        # (x_lo, x_hi, y_lo, y_hi, color, name, text_color, edge_color, fc_box)
        (0,   xt,   yt,   ymax, "#2ecc71", "DOBRY",
         "#1a7a45", "#2ecc71", "white"),
        (xt,  xmax, yt,   ymax, "#f39c12", "Przewidywalnie\nsłaby",
         "#7d4e00", "#f39c12", "white"),
        (0,   xt,   ymin, yt,   "#c0392b", "NAJGORSZY",
         "#922b21", "#c0392b", "#fdecea"),
        (xt,  xmax, ymin, yt,   "#e67e22", "Zły",
         "#7d3c00", "#e67e22", "white"),
    ]

    def quadrant_idx(xi, yi):
        low  = xi <  xt
        norm = yi >= yt
        if low  and norm: return 0
        if not low and norm: return 1
        if low  and not norm: return 2
        return 3

    colors_per_point = [QUADRANTS[quadrant_idx(xi, yi)][3] for xi, yi in zip(x, y)]
    counts = [sum(quadrant_idx(xi, yi) == k for xi, yi in zip(x, y)) for k in range(4)]

    fig, ax = plt.subplots(figsize=(9, 7))

    # ── Background shading ────────────────────────────────────────────────
    for x_lo, x_hi, y_lo, y_hi, color, *_ in QUADRANTS:
        ax.add_patch(Rectangle(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            facecolor=color, alpha=0.09, zorder=0,
        ))

    # ── Threshold lines ───────────────────────────────────────────────────
    ax.axvline(xt, color="black", lw=1.8, ls="--", alpha=0.6, zorder=2)
    ax.axhline(yt, color="black", lw=1.8, ls="--", alpha=0.6, zorder=2)

    # ── Scatter ───────────────────────────────────────────────────────────
    ax.scatter(x, y, c=colors_per_point, s=55, alpha=0.75,
               edgecolors="white", linewidths=0.5, zorder=5)

    # ── Liczniki w rogach kwadrantów ──────────────────────────────────────
    # (lewy/prawy róg, góra/dół — zależnie od kwadrantu)
    badge_y_top = np.exp(np.log(yt) + (np.log(ymax) - np.log(yt)) * 0.93)
    badge_y_bot = np.exp(np.log(ymin) + (np.log(yt) - np.log(ymin)) * 0.07)
    corners = [
        (xt * 0.04,          badge_y_top, "left",  "top"),     # Q1 góra-lewo
        (xt + (xmax-xt)*0.04, badge_y_top, "left", "top"),     # Q2 góra-lewo
        (xt * 0.04,          badge_y_bot, "left",  "bottom"),  # Q3 dół-lewo
        (xt + (xmax-xt)*0.04, badge_y_bot, "left", "bottom"),  # Q4 dół-lewo
    ]
    for (q_def, corner, cnt) in zip(QUADRANTS, corners, counts):
        _, _, _, _, color, _, txt_color, _, _ = q_def
        cx, cy, ha, va = corner
        ax.text(cx, cy, f"n = {cnt}  ({cnt / n * 100:.0f}%)",
                ha=ha, va=va, fontsize=11,
                color=txt_color, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color,
                          alpha=0.88, lw=1.4))

    # ── Adnotacje progów ──────────────────────────────────────────────────
    ax.text(xmax * 0.99, yt * 1.4,
            f"p = {yt}", fontsize=10, color="gray", ha="right")

    ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("RMSSE", fontsize=14)
    ax.set_ylabel("Mediana p-wartości Shapiro-Wilka  [skala log]", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, which="both", alpha=0.15, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = output_dir / f"reliability_quadrant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_reliability_bias_profiles(df, param_names, bounds, output_dir: Path):
    """Per-parameter bias profile: mean bias ± 1 SD vs true parameter value.

    Each dot = one LHS scan point (mean over n_repeats fits).
    Red line = LOWESS trend. Blue band = ±1 SD (smoothed).
    """
    n_params = len(param_names)
    fig_w = max(10, 2.8 * n_params + 1)
    fig, axes = plt.subplots(1, n_params, figsize=(fig_w, 5))
    if n_params == 1:
        axes = [axes]

    for i, (name, (lo, hi)) in enumerate(zip(param_names, bounds)):
        ax = axes[i]
        x    = df[f"true_{name}"].values
        bias = df[f"bias_{name}"].values
        std  = df[f"std_{name}"].values

        sort_idx = np.argsort(x)
        xs, bs, ss = x[sort_idx], bias[sort_idx], std[sort_idx]

        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sm_bias = lowess(bs, xs, frac=0.5)[:, 1]
            sm_hi   = lowess(bs + ss, xs, frac=0.5)[:, 1]
            sm_lo   = lowess(bs - ss, xs, frac=0.5)[:, 1]
            ax.fill_between(xs, sm_lo, sm_hi, alpha=0.18, color="#3498db", zorder=1)
            ax.plot(xs, sm_bias, color="#e74c3c", lw=2.2, zorder=4)
        except ImportError:
            ax.fill_between(xs, bs - ss, bs + ss, alpha=0.15, color="#3498db", zorder=1)

        ax.scatter(x, bias, alpha=0.35, s=18, color="#3498db",
                   linewidths=0, zorder=2)
        ax.axhline(0, color="black", lw=1.3, ls="--", alpha=0.5, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlabel("prawdziwa wartość parametru", fontsize=13)
        ax.tick_params(labelsize=11)
        if i == 0:
            ax.set_ylabel(r"Obciążenie  $\hat{\theta} - \theta$", fontsize=13)
        ax.grid(True, alpha=0.2, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = output_dir / f"reliability_bias_profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def _add_lowess(ax, x, y, color="red", lw=2):
    """Fit and draw a LOWESS trend line."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        xy_sorted = sorted(zip(x, y))
        xs, ys = zip(*xy_sorted)
        smoothed = lowess(ys, xs, frac=0.5)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, lw=lw)
    except ImportError:
        m = np.polyfit(x, y, 1)
        xr = np.linspace(min(x), max(x), 50)
        ax.plot(xr, np.poly1d(m)(xr), color=color, lw=lw, ls="--")
