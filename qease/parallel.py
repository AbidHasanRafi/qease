"""Multi-GPU parallel map used by every stage."""
import queue
import threading
import traceback

from tqdm.auto import tqdm
import torch
from qease import state
from qease.utils import log


def parallel_map(items, fn, desc="", runners=None, total=None):
    """Run fn(runner, item) over items using one worker thread per GPU replica. Errors are logged, not raised."""
    runners = runners or state.RUNNERS
    q = queue.Queue()
    for it in items: q.put(it)
    results, lock = [], threading.Lock()
    bar = tqdm(total=total or len(items), desc=desc)
    def tick(n=1): bar.update(n)
    def worker(r):
        if r.device.startswith("cuda"): torch.cuda.set_device(r.device)
        with torch.inference_mode():
            while not state.STOP.is_set():
                try: it = q.get_nowait()
                except queue.Empty: return
                try:
                    out = fn(r, it, tick)
                    with lock: results.append(out)
                except Exception as e:
                    with lock: state.FAILS.append((desc, str(it)[:80], repr(e)[:200], traceback.format_exc()))
                    if "out of memory" in str(e).lower() and r.device.startswith("cuda"):
                        torch.cuda.empty_cache()
    ths = [threading.Thread(target=worker, args=(r,), daemon=True) for r in runners]
    for t in ths: t.start()
    try:
        for t in ths:
            while t.is_alive(): t.join(0.5)
    except KeyboardInterrupt:
        state.STOP.set(); log("interrupted - waiting for workers to finish current item")
        for t in ths: t.join()
        state.STOP.clear(); raise
    bar.close()
    if state.FAILS and state.FAILS[-1][0] == desc:
        n_f = sum(1 for f in state.FAILS if f[0] == desc)
        log(f"[{desc}] {n_f} item(s) failed -> see FAILS[-1][3] for the last traceback")
    return results
