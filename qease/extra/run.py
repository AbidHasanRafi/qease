"""One pass of the extra experiments for one model."""
import gc
import math
import random
import threading
import traceback

import torch
from qease import state
from qease.extra import xstate
from qease.crc import load_eval
from qease.data import interleave_by_ds
from qease.parallel import parallel_map
from qease.selector import load_probe_records
from qease.extra.benchmarks import x_bench_one, x_bench_recs, x_bench_report, x_std_build
from qease.extra.question_swap import x_qswap_shard
from qease.extra.reports import x_qswap_report, x_tb_report
from qease.extra.trained_baselines import X_KIND, x_attn_shard, x_load_tb, x_tbeval_shard, x_train
from qease.extra.utils import xd, xlog


def x_run_pass(label, do_tb, do_qswap, do_bench, n_bench, do_std=False, n_std=0):
    # run-time state lives in qease.extra.xstate
    xlog(f"=== pass '{label}'  model={state.CFG['MODEL_ID']}  work={state.WORK} ===")
    for _r in state.RUNNERS: _r.vz_on = bool(state.VZ_OK)                                   # VisionZip needs vision attention during prepare()
    xstate.UID2REC = {r["uid"]: r for r in state.POOL}
    x_load_tb()
    test_uids = [r["uid"] for r in load_eval("test")] or [r["uid"] for r in state.BY_SPLIT["test"]]
    test_recs = [xstate.UID2REC[u] for u in test_uids if u in xstate.UID2REC]
    # ---------------- A  trained baselines
    if do_tb:
        PROBE = load_probe_records(); assert PROBE, "probe shards missing - copy the first run's WORK/probe folder"
        if not all(n in state.SELECTORS for n in xstate.XCFG["TB_NAMES"]):
            A_map = {}
            if "attn" in xstate.XCFG["TB_NAMES"] and "attn" not in state.SELECTORS:
                if state.FASTV_OK:
                    uids = [p["uid"] for p in PROBE]
                    shards = [(i, [xstate.UID2REC[u] for u in uids[i * 100:(i + 1) * 100] if u in xstate.UID2REC]) for i in range(math.ceil(len(uids) / 100))]
                    todo = [s for s in shards if not (xd("attn") / f"shard_{s[0]:04d}.pt").exists()]
                    xlog(f"[tb] FastV attention targets for {len(uids)} training images ({len(todo)} shards to do)")
                    if todo: parallel_map(todo, x_attn_shard, "tb-attn", total=sum(len(s[1]) for s in todo))
                    for p in sorted(xd("attn").glob("shard_*.pt")): A_map.update(torch.load(p, weights_only=False))
                else:
                    xlog("[tb] FastV self-test failed -> the attention-distilled baseline is skipped")
            todo_names = [n for n in xstate.XCFG["TB_NAMES"] if n not in state.SELECTORS and not (n == "attn" and not A_map)]
            for i in range(0, len(todo_names), max(1, len(state.DEVICES))):
                chunk, res, ths = todo_names[i:i + len(state.DEVICES)], {}, []
                def _tw(n, dev):
                    if dev.startswith("cuda"): torch.cuda.set_device(dev)
                    try: res[n] = x_train(n, X_KIND[n], dev, PROBE, A_map)
                    except Exception as e: state.FAILS.append(("tb-train", n, repr(e)[:200], traceback.format_exc())); xlog(f"training '{n}' failed: {repr(e)[:160]}")
                for j, n in enumerate(chunk):
                    t = threading.Thread(target=_tw, args=(n, state.DEVICES[j % len(state.DEVICES)]), daemon=True); t.start(); ths.append(t)
                for t in ths:
                    while t.is_alive(): t.join(0.5)
                state.SELECTORS.update(res)
        del PROBE; gc.collect()
        S = 25; shards = [(i, test_recs[i * S:(i + 1) * S]) for i in range(math.ceil(len(test_recs) / S))]
        todo = [s for s in shards if not (xd("tbeval") / f"test_{s[0]:04d}.pt").exists()]
        xlog(f"[tb] evaluating {[n for n in state.SELECTORS]} on {len(test_recs)} test questions ({len(todo)} shards)")
        if todo: parallel_map(todo, x_tbeval_shard, "tb-eval", total=sum(len(s[1]) for s in todo))
        x_tb_report()
    # ---------------- B  question swap
    if do_qswap:
        pick = interleave_by_ds(test_recs, xstate.XCFG["QSWAP_N"], seed=11); rng = random.Random(5); pairs = []
        for rec in pick:
            others = [o for o in test_recs if o["ds"] == rec["ds"] and o["uid"] != rec["uid"] and o["question"].strip().lower() != rec["question"].strip().lower()]
            pairs.append((rec, rng.choice(others)))
        shards = [(i, pairs[i * 10:(i + 1) * 10]) for i in range(math.ceil(len(pairs) / 10))]
        todo = [s for s in shards if not (xd("qswap") / f"shard_{s[0]:04d}.pt").exists()]
        xlog(f"[qswap] {len(pairs)} (image, own question, other question) triples, {len(todo)} shards")
        if todo: parallel_map(todo, x_qswap_shard, "qswap", total=sum(len(s[1]) for s in todo))
        x_qswap_report()
    # ---------------- C  downstream accuracy
    if do_bench:
        recs = x_bench_recs(n_bench)
        todo = [r for r in recs if not (xd("bench") / f"{r['uid']}.json").exists()]
        xlog(f"[bench] {len(recs)} test questions ({n_bench}/dataset), {len(todo)} to do | FASTV_OK={state.FASTV_OK} VZ_OK={state.VZ_OK} SV_OK={state.SV_OK}")
        if todo: parallel_map(todo, x_bench_one, "bench", total=len(todo))
        x_bench_report()
    # ---------------- E  standard benchmarks (official validation / test sets disjoint from the training pool)
    if do_std:
        state.CFG["TOKENS_PER_DS"]["std_textvqa"] = state.CFG["TOKENS_PER_DS"].get("textvqa", 400)
        recs = []
        for nm in xstate.XCFG["STD_NAMES"]: recs += x_std_build(nm, n_std)
        todo = [r for r in recs if not (xd("stdbench") / f"{r['uid']}.json").exists()]
        xlog(f"[std] {len(recs)} questions from {xstate.XCFG['STD_NAMES']} ({n_std} each), {len(todo)} to do")
        if todo: parallel_map(todo, x_bench_one, "std-bench", total=len(todo))
        x_bench_report("stdbench", [f"std_{n}" for n in xstate.XCFG["STD_NAMES"]], "std_bench")
    if state.FAILS: xlog(f"{len(state.FAILS)} item(s) failed in total; last: {state.FAILS[-1][0]} {state.FAILS[-1][1]} {state.FAILS[-1][2]}  (full traceback: FAILS[-1][3])")
