"""Analysis block A15: manifest, README of the results folder and the results archive."""
import matplotlib
matplotlib.use("Agg")                      # figures are written to disk
import time

import pandas as pd
from qease import state
from qease.utils import log


def an_finalize():
    state.MANIFEST.append(("meta", "results/README_results.md", "this index")); state.MANIFEST.append(("meta", "results/MANIFEST.csv", "machine-readable index"))
    M = pd.DataFrame(state.MANIFEST, columns=["kind", "path", "description"]); M.to_csv(state.WORK / "results" / "MANIFEST.csv", index=False)
    lines = [f"# Q-EASE results ({state.TAG})", "", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Test n={len(state.TEST)}, calibration n={len(state.CAL)}.", "",
             "* Every figure has `figures/<name>.pdf|png` (vector PDF, TrueType fonts; 600-dpi PNG) and the exact tidy data in `figdata/<name>.csv`.",
             "* Every table is a CSV (+ a `.tex` booktabs-style export when pandas/jinja2 allow).",
             f"* All uncertainty = 95% percentile bootstrap over test samples (B={state.CFG['BOOT_B']}) unless stated; risks use Wilson intervals.",
             "* CRC guarantee is marginal (calibration+test exchangeable); see `conditional_validity` and `crc_validity_resampled` for how it behaves empirically.", ""]
    for kind in ["figure", "figure-data", "table", "meta"]:
        lines += [f"## {kind}", ""] + [f"* `{r.path}` - {r.description}" for r in M[M.kind == kind].itertuples()] + [""]
    (state.WORK / "results" / "README_results.md").write_text("\n".join(lines))
    (state.WORK / "results" / "recreate_figure_example.py").write_text(
        '"""Re-draw a curve figure from its exported data (no GPU / model needed)."""\nimport pandas as pd, matplotlib.pyplot as plt\n'
        'd = pd.read_csv("figdata/fig_fidelity_curves.csv")\nfig, ax = plt.subplots(figsize=(3.5, 2.6))\n'
        'for m, g in d[d.metric == "flip"].groupby("method"):\n    g = g.sort_values("frac"); ax.plot(g.frac, g["mean"], marker="o", label=m); ax.fill_between(g.frac, g.lo, g.hi, alpha=.15)\n'
        'ax.set_xscale("log", base=2); ax.set_xlabel("fraction of groups kept"); ax.set_ylabel("answer-flip rate"); ax.legend(fontsize=6)\nfig.savefig("redrawn.pdf", bbox_inches="tight")\n')
    import shutil
    z = shutil.make_archive(str(state.BASE / f"qease_results_{state.TAG}"), "zip", state.WORK / "results")
    log(f"[analyze] results zipped -> {z}")
