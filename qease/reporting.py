"""Plot style, table / figure export helpers shared by the analysis."""
import matplotlib
matplotlib.use("Agg")                      # figures are written to disk
import math

from fractions import Fraction
import matplotlib.pyplot as plt
from qease import state


COL1, COL2 = 3.5, 7.16                           # IEEE / ACM single- and double-column widths (inches)


LN2 = math.log(2)


DS_LABEL = dict(vqav2="VQAv2", gqa="GQA", textvqa="TextVQA", docvqa="DocVQA", aokvqa="A-OKVQA", vsr="VSR", clevr="CLEVR")


METHOD_STYLE = {                                  # Okabe-Ito colour-blind-safe palette, distinct markers + line styles (B/W safe)
    "qease":     dict(color="#D55E00", marker="o", ls="-",  label="Q-EASE (ours)"),
    "qease_noq": dict(color="#CC79A7", marker="s", ls="--", label="Q-EASE w/o question"),
    "qsim":      dict(color="#0072B2", marker="^", ls="-.", label="Question-similarity top-$k$"),
    "uniform":   dict(color="#009E73", marker="D", ls=":",  label="Uniform grid"),
    "random":    dict(color="#7F7F7F", marker="v", ls="-",  label="Random"),
    "oracle1":   dict(color="#000000", marker="*", ls="-",  label="Single-ablation oracle"),
    "fastv":     dict(color="#E69F00", marker="P", ls="-",  label="FastV ($K$=2, re-impl.)"),
    "visionzip": dict(color="#56B4E9", marker="X", ls="-",  label="VisionZip (port)"),
    "sparsevlm": dict(color="#8C564B", marker="h", ls="-",  label="SparseVLM v1.0 (port)"),
}


ALL_BASELINES = ["fastv", "visionzip", "qsim", "uniform", "random"]


BASELINES = list(ALL_BASELINES)                  # narrowed in prepare_analysis_state() to what is actually in the eval records


_EXTRA_COLORS = ["#56B4E9", "#F0E442", "#4D4D4D", "#999933"]


POLICY_LABEL = dict(fixed="fixed-$k$", adaptive="adaptive (CRC)", mondrian="adaptive (per-dataset CRC)", naive="adaptive (uncalibrated)")


def set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,       # Origin-style boxed axes
        "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "axes.axisbelow": True, "legend.frameon": True, "legend.edgecolor": "0.3", "legend.fancybox": False,
        "legend.framealpha": 0.92, "legend.borderpad": 0.3, "legend.handlelength": 1.8, "legend.labelspacing": 0.25,
        "figure.dpi": 120, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",                                 # embeddable TrueType (IEEE/ACM requirement)
    })


def save_table(df, name, desc, latex=True):
    p = state.TAB / f"{name}.csv"; df.to_csv(p, index=False)
    state.MANIFEST.append(("table", str(p.relative_to(state.WORK)), desc + f"  [columns: {', '.join(map(str, df.columns))}]"))
    if latex:
        try: (state.TAB / f"{name}.tex").write_text(df.to_latex(index=False, float_format=lambda x: f"{x:.4f}", escape=True))
        except Exception: pass
    return df


def save_fig(fig, name, data, desc, show=True):
    """PDF (vector) + PNG (600 dpi) + the exact tidy data that was plotted."""
    for ext in ("pdf", "png"):
        fig.savefig(state.FIG / f"{name}.{ext}")
    data.to_csv(state.FDATA / f"{name}.csv", index=False)
    state.MANIFEST.append(("figure", f"results/figures/{name}.pdf|png", desc))
    state.MANIFEST.append(("figure-data", f"results/figdata/{name}.csv", f"data behind {name}  [columns: {', '.join(map(str, data.columns))}]"))
    plt.close(fig)
    if show:
        try:
            from IPython.display import Image as IPImage, display
            display(IPImage(filename=str(state.FIG / f"{name}.png"), width=760))
        except Exception:
            pass


def panel_label(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, fontweight="bold", va="top", ha="left", fontsize=8)


def frac_ticks(ax, fr):
    ax.set_xscale("log", base=2)
    ax.set_xticks(fr); ax.set_xticklabels([str(Fraction(float(f)).limit_denominator(32)) for f in fr])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


set_style()
