"""
plots/comparison.py
===================
Visualisations for the cross-fitting comparison analysis.

Three main figures:
  1. BIC heatmap  — which model wins on whose data
  2. NLL violin   — fit quality distribution per (data, fit) pair
  3. Recovery bar — within-model RMSSE side-by-side
  4. Model confusion matrix — % of times true model wins BIC
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def plot_cross_fit_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate all cross-fit comparison figures.

    Call after run_cross_fit_comparison().
    """
    from analysis.comparison import summarize_cross_fit, best_fitting_model_per_agent

    summary = summarize_cross_fit(df)
    best_df = best_fitting_model_per_agent(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_bic_heatmap(summary, output_dir)
    _plot_nll_violin(df, output_dir)
    _plot_rmsse_comparison(summary, output_dir)
    _plot_model_confusion(best_df, output_dir)
    _plot_bic_delta(df, output_dir)

    print(f"\nAll comparison plots saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Figure 1 — BIC heatmap
# ---------------------------------------------------------------------------

def _plot_bic_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    """Pivot BIC means into a heatmap: rows = data_model, cols = fit_model.

    Diagonal = self-fit (parameter recovery scenario).
    Off-diagonal = cross-fit (model mismatch scenario).
    Lower BIC = better fit.

    The cell colour is the DELTA-BIC relative to the best model for each row,
    so we see: "how much worse is each fit_model compared to the winner?"
    """
    pivot = summary.pivot(index="data_model", columns="fit_model", values="bic_mean")

    # Delta-BIC: subtract row minimum so the best model = 0
    delta = pivot.subtract(pivot.min(axis=1), axis=0)

    models = list(pivot.index)
    n = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(7 * n, 4 * n), squeeze=False)
    axes = axes[0]

    # ── Left: raw BIC ────────────────────────────────────────────────────────
    mask_diag = np.eye(n, dtype=bool)
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="YlOrRd_r",
        linewidths=0.5, ax=axes[0],
        cbar_kws={"label": "Mean BIC"},
    )
    # Bold the diagonal
    for i in range(n):
        axes[0].add_patch(plt.Rectangle((i, i), 1, 1,
                          fill=False, edgecolor="black", lw=3))
    axes[0].set_title("Mean BIC  (lower = better fit)\nDiagonal = self-fit",
                      fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Fitting model", fontsize=11)
    axes[0].set_ylabel("Data-generating model", fontsize=11)

    # ── Right: ΔBIC ─────────────────────────────────────────────────────────
    sns.heatmap(
        delta, annot=True, fmt=".1f", cmap="Reds",
        linewidths=0.5, ax=axes[1],
        cbar_kws={"label": "ΔBIC vs best model in row"},
        vmin=0,
    )
    for i in range(n):
        axes[1].add_patch(plt.Rectangle((i, i), 1, 1,
                          fill=False, edgecolor="black", lw=3))
    axes[1].set_title("ΔBIC vs best fit_model per row\n0 = winner", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Fitting model", fontsize=11)
    axes[1].set_ylabel("")

    plt.tight_layout()
    fname = output_dir / f"crossfit_bic_heatmap_{_ts()}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Figure 2 — NLL violin plots
# ---------------------------------------------------------------------------

def _plot_nll_violin(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin plot of NLL distributions per (data_model, fit_model) pair.

    Each panel = one data-generating model.
    X-axis = fitting models.
    The violin for the TRUE model should be leftmost (lowest NLL) and narrowest.
    """
    data_models = df["data_model"].unique()
    n = len(data_models)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, dgc_name in zip(axes, data_models):
        sub = df[df["data_model"] == dgc_name]
        fit_models = sorted(sub["fit_model"].unique())

        # Colour: green for self-fit, grey for others
        palette = {
            fm: "#27ae60" if fm == dgc_name else "#95a5a6"
            for fm in fit_models
        }

        sns.violinplot(
            data=sub, x="fit_model", y="nll",
            palette=palette, ax=ax,
            order=fit_models,
            inner="quartile", cut=0,
        )
        ax.set_title(f"Data from: {dgc_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fitting model", fontsize=10)
        ax.set_ylabel("NLL" if dgc_name == data_models[0] else "")
        ax.tick_params(axis="x", rotation=30)

        # Horizontal line at median NLL of self-fit
        self_median = sub[sub["fit_model"] == dgc_name]["nll"].median()
        ax.axhline(self_median, color="#27ae60", ls="--", lw=1.5,
                   label=f"Self-fit median = {self_median:.1f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fname = output_dir / f"crossfit_nll_violin_{_ts()}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Figure 3 — Within-model RMSSE comparison (parameter recovery)
# ---------------------------------------------------------------------------

def _plot_rmsse_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    """Side-by-side bar chart of within-model RMSSE (median and p90).

    This is the apples-to-apples comparison: each model on its own data.
    Lower RMSSE = better parameter recovery.
    """
    self_fit = summary[summary["data_model"] == summary["fit_model"]].copy()
    self_fit = self_fit.sort_values("rmsse_median")

    fig, ax = plt.subplots(figsize=(max(8, len(self_fit) * 2), 6))

    x = np.arange(len(self_fit))
    width = 0.35

    bars1 = ax.bar(x - width/2, self_fit["rmsse_median"],
                   width, label="Median RMSSE", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width/2, self_fit["rmsse_p90"],
                   width, label="P90 RMSSE", color="#e74c3c", alpha=0.85)

    # Annotate values
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(self_fit["data_model"], fontsize=11)
    ax.set_ylabel("RMSSE (scaled)", fontsize=11)
    ax.set_ylim(0, max(self_fit["rmsse_p90"].max() * 1.25, 0.5))
    ax.axhline(0.2, color="gray", ls="--", lw=1, alpha=0.6, label="Threshold 0.2")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fname = output_dir / f"crossfit_rmsse_comparison_{_ts()}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Figure 4 — Model confusion matrix
# ---------------------------------------------------------------------------

def _plot_model_confusion(best_df: pd.DataFrame, output_dir: Path) -> None:
    """Confusion matrix: for data from model X, which model wins BIC (% of agents)?

    Rows = data_model, columns = best_fit_model (winner by BIC).
    Diagonal = % of times the TRUE model wins — should be high.
    Off-diagonal = model confusion rate — THIS IS THE CLINICAL DANGER SIGNAL.

    A high off-diagonal entry means: "X% of participants whose data was generated
    by Model A are better explained by Model B — meaning we would attribute the
    WRONG psychological processes to them."
    """
    data_models = sorted(best_df["data_model"].unique())
    fit_models  = sorted(best_df["best_fit_model"].unique())
    all_models  = sorted(set(data_models) | set(fit_models))

    # Build confusion matrix (rows = data, cols = best_fit)
    confusion = pd.DataFrame(0.0, index=all_models, columns=all_models)
    for (dgc_name,), grp in best_df.groupby(["data_model"]):
        total = len(grp)
        for fm, count in grp["best_fit_model"].value_counts().items():
            confusion.loc[dgc_name, fm] = count / total * 100

    fig, ax = plt.subplots(figsize=(max(8, len(all_models) * 2.5),
                                    max(6, len(all_models) * 2.2)))

    annot = confusion.copy().round(1).astype(str) + "%"
    sns.heatmap(
        confusion, annot=annot, fmt="s",
        cmap="Blues", vmin=0, vmax=100,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "% agents where this model wins BIC"},
    )

    # Red border on diagonal cells
    n = len(all_models)
    for i, model in enumerate(all_models):
        if model in confusion.index and model in confusion.columns:
            col_idx = list(confusion.columns).index(model)
            row_idx = list(confusion.index).index(model)
            ax.add_patch(plt.Rectangle(
                (col_idx, row_idx), 1, 1,
                fill=False, edgecolor="#27ae60", lw=3
            ))

    ax.set_xlabel("Model that WINS BIC", fontsize=12)
    ax.set_ylabel("True data-generating model", fontsize=12)

    plt.tight_layout()
    fname = output_dir / f"crossfit_confusion_{_ts()}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

    # Print summary
    print("\n=== MODEL CONFUSION SUMMARY ===")
    for model in all_models:
        if model in confusion.index:
            win_rate = confusion.loc[model, model] if model in confusion.columns else 0
            print(f"  {model}: correct model wins in {win_rate:.1f}% of cases")


# ---------------------------------------------------------------------------
# Figure 5 — ΔBIC distribution (clinical danger quantification)
# ---------------------------------------------------------------------------

def _plot_bic_delta(df: pd.DataFrame, output_dir: Path) -> None:
    """For each (data_model, agent), compute BIC(true_model) - BIC(best_competitor).

    Positive ΔBIC = true model is WORSE than the competitor = misclassification.
    Negative ΔBIC = true model wins correctly.

    This directly quantifies the clinical risk: how often and by how much
    does the wrong model provide a better fit?
    """
    data_models = df["data_model"].unique()
    n = len(data_models)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, dgc_name in zip(axes, data_models):
        sub = df[df["data_model"] == dgc_name].copy()

        # For each sample, compute: BIC_true - BIC_best_competitor
        records = []
        for sample_idx, grp in sub.groupby("sample_idx"):
            bic_per = grp.groupby("fit_model")["bic"].mean()
            if dgc_name not in bic_per.index:
                continue
            bic_true = bic_per[dgc_name]
            competitors = bic_per.drop(index=dgc_name)
            if len(competitors) == 0:
                continue
            bic_best_comp = competitors.min()
            records.append(bic_true - bic_best_comp)

        if not records:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        delta_bic = np.array(records)
        pct_misclassified = (delta_bic > 0).mean() * 100

        color = "#e74c3c" if pct_misclassified > 20 else "#27ae60"

        ax.hist(delta_bic, bins=30, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", lw=2, ls="--",
                   label="ΔBIC = 0\n(true = competitor)")
        ax.axvline(np.median(delta_bic), color="navy", lw=1.5,
                   label=f"Median = {np.median(delta_bic):.1f}")

        ax.set_title(
            f"Data from: {dgc_name}\n"
            f"Misclassified: {pct_misclassified:.1f}%",
            fontsize=11, fontweight="bold",
            color="#e74c3c" if pct_misclassified > 20 else "black",
        )
        ax.set_xlabel("ΔBIC  (true model − best competitor)", fontsize=10)
        ax.set_ylabel("Count" if dgc_name == data_models[0] else "")
        ax.legend(fontsize=8)

        # Shade misclassification zone
        xlim = ax.get_xlim()
        ax.axvspan(0, max(xlim[1], 1), alpha=0.08, color="#e74c3c",
                   label="Wrong model wins")

    plt.tight_layout()
    fname = output_dir / f"crossfit_bic_delta_{_ts()}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _ts() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")