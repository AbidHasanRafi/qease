"""Q-EASE - escalate-or-abstain token pruning for lightweight VLMs.   Run the whole experiment:   python main.py

No configuration lives here.  The defaults are the full experiment (identical to the one-cell script `qease_final.py`);
settings are in qease/parts/01_config.py, or pass JSON through the QEASE_OVERRIDES environment variable (see README.md).

How it works: the pipeline is ONE program split into ordered parts (qease/parts/00_*.py ... 19_*.py).  The parts are executed one after the
other in a SINGLE shared namespace (the module `qease_pipeline`), exactly as if they were pasted into one notebook cell - there is deliberately no
main() function: every stage is idempotent and checkpoints to disk, so re-running `python main.py` resumes where a crashed run stopped, and
after a crash (or in a notebook via `%run main.py`) all state stays inspectable:  sys.modules["qease_pipeline"].FAILS[-1]
"""
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                                        # off Kaggle the outputs go to ./qease_out next to the code

pipeline = types.ModuleType("qease_pipeline")         # the shared namespace (a real module so that pickling of classes defined in the parts works)
pipeline.__file__ = str(ROOT / "main.py")
sys.modules["qease_pipeline"] = pipeline

for part in sorted((ROOT / "qease" / "parts").glob("*.py")):
    print(f"\n######## {part.name} ########", flush=True)
    code = compile(part.read_text(encoding="utf-8"), str(part), "exec")      # file name kept -> tracebacks point at the right part and line
    try:
        exec(code, pipeline.__dict__)
    except BaseException:
        print(f"\n[main.py] stopped inside {part.name}; fix it and run `python main.py` again (finished stages are skipped).", flush=True)
        raise
