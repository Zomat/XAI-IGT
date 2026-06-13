from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

_PALETTE = {
    "GOB": "#2ecc71",
    "BOG": "#e74c3c",
    "IOF": "#3498db",
    "FOI": "#f1c40f",
    "Remaining": "#95a5a6",
}


def plot_psp_results(df, mode_name: str, output_dir: Path):
    """Pairplot of parameter space colored by behavioral pattern."""
    g = sns.pairplot(
        df, hue="Pattern", palette=_PALETTE, corner=True,
        plot_kws={"alpha": 0.3, "edgecolor": None, "s": 5},
    )

    # corner=True hides y-axis on diagonal KDE — force label via annotate
    param_cols = [c for c in df.columns if c != "Pattern"]
    ax0 = g.axes[0, 0]
    if ax0 is not None:
        ax0.annotate(param_cols[0],
                     xy=(0, 0.5), xycoords="axes fraction",
                     xytext=(-0.18, 0.5), textcoords="axes fraction",
                     fontsize=10, ha="right", va="center", rotation=90)

    # Legend: larger font, visible frame, Polish title
    legend = g._legend
    if legend is None:
        for ax_row in g.axes:
            for ax in ax_row:
                if ax is not None and ax.get_legend() is not None:
                    legend = ax.get_legend()
                    break
    if legend:
        legend.set_title("Wzorzec")
        legend.get_title().set_fontsize(16)
        for text in legend.get_texts():
            text.set_fontsize(14)
        for handle in legend.legend_handles:
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([80])
            elif hasattr(handle, "set_markersize"):
                handle.set_markersize(12)
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#888888")
        frame.set_linewidth(1.5)

    _mode_pl = {"Broad": "Szeroki", "Restricted": "Zawężony"}.get(mode_name, mode_name)
    g.figure.subplots_adjust(top=0.93)
    fname = output_dir / f"psp_pairplot_{mode_name.lower()}.png"
    g.figure.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_psp_distribution(percents, mode_name: str, output_dir: Path):
    """Bar chart of behavioral pattern proportions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=percents.index, y=percents.values, palette=_PALETTE, ax=ax)
    _mode_pl = {"Broad": "Szeroki", "Restricted": "Zawężony"}.get(mode_name, mode_name)
    ax.set_ylabel("Udział w przestrzeni parametrów [%]")
    ax.set_xlabel("Fenotyp (Steingroever i in.)")
    ax.set_ylim(0, 100)
    for i, p in enumerate(percents.values):
        ax.text(i, p + 1, f"{p:.1f}%", ha="center", fontweight="bold")

    fname = output_dir / f"psp_dist_{mode_name.lower()}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
