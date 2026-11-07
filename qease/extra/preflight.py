"""Copies the first run's outputs (selectors, thresholds, evaluation records) into the working directory when they are missing."""
from pathlib import Path
from qease.extra import xstate
from qease.extra.utils import xlog


def x_preflight(model_id, base_dir=None):
    """the first run's outputs must be in /kaggle/working/qease; if not, find them in /kaggle/input and copy them (otherwise the main pipeline
    would silently start rebuilding the data pool from the internet)."""
    import shutil
    base = Path(base_dir) if base_dir else (Path("/kaggle/working/qease") if Path("/kaggle/working").exists() else Path("./qease_out"))
    tag = model_id.rstrip("/").split("/")[-1]
    need = [base / tag / "ckpt" / "selector_full.pt", base / tag / "results" / "tables" / "lambdas.json", base / tag / "eval"]   # ONLY the first run can provide these
    optional = [base / "pool" / "vqav2.done", base / tag / "probe"]        # the pool is re-downloaded from HF if absent; probe shards only feed block A

    def _report():
        if not (base / "pool" / "vqav2.done").exists():
            xlog("NOTE: image pool not found -> the main pipeline will REBUILD it from Hugging Face (~20-40 min, internet must be ON)")
        if not list((base / tag / "probe").glob("shard_*.pt")):
            if xstate.XCFG["RUN_TB"]: xlog("NOTE: probe shards not found -> trained-baselines block A is switched OFF (needs the first run's probe/ folder)")
            xstate.XCFG["RUN_TB"] = False
    miss = [p for p in need if not p.exists()]
    if not miss:
        xlog("preflight ok: first-run outputs found in", base); _report(); return
    xlog("preflight: missing in the working dir:", [str(p) for p in miss])
    found = None
    if xstate.XCFG.get("QEASE_SRC") and (Path(xstate.XCFG["QEASE_SRC"]) / tag / "ckpt" / "selector_full.pt").exists():
        found = Path(xstate.XCFG["QEASE_SRC"])
    if found is None and Path("/kaggle/input").exists():
        xlog("searching /kaggle/input for the first run's 'qease' folder ...")
        for p in Path("/kaggle/input").glob("**/selector_full.pt"):
            if p.parent.name == "ckpt" and p.parent.parent.name == tag: found = p.parent.parent.parent; break
    if found is None:
        raise SystemExit(f"first-run outputs not found.  Expected {base}/{tag}/ckpt/selector_full.pt, {base}/{tag}/results/tables/lambdas.json and {base}/{tag}/eval/.\n"
                         f"These come from YOUR first run (trained selectors, calibrated thresholds, test split) and cannot be downloaded.  Minimum upload (~60 MB), keeping this layout:\n"
                         f"  qease/{tag}/ckpt/   qease/{tag}/eval/   qease/{tag}/results/tables/lambdas.json\n"
                         f"(add qease/pool/ [1.6 GB] to skip the HF re-download, and qease/{tag}/probe/ [1.4 GB] to enable the trained-baselines block).  Add it as a Kaggle Dataset input and re-run.")
    xlog(f"copying {found} -> {base}  (one-off, a few minutes) ...")
    shutil.copytree(found, base, dirs_exist_ok=True)
    miss = [p for p in need if not p.exists()]
    if miss: raise SystemExit(f"still missing after the copy: {[str(p) for p in miss]}  (lambdas.json needs the 'crc' stage of the first run)")
    xlog("preflight ok after copy"); _report()
