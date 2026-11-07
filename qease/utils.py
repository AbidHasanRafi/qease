"""Logging, environment check and the deterministic hashing used for the image-level splits."""
import hashlib
import os
import subprocess
import sys
import time
import warnings


warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")


def check_environment(min_transformers="4.45.0"):
    """Fail early (with an actionable message) if transformers is too old for Qwen2-VL / Qwen2.5-VL."""
    try:
        import transformers
        from packaging import version as _v
        if _v.parse(transformers.__version__) < _v.parse(min_transformers):
            _pip("-U", f"transformers>={min_transformers}")
            raise ImportError("restart the kernel after upgrading transformers")
    except ImportError as _e:
        raise SystemExit(f"{_e}  -> restart the kernel and run again.")


def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stable_hash(s):
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)


def split_of(img_key):
    """Deterministic split by IMAGE so no image leaks across train/val/cal/test."""
    h = stable_hash(img_key) % 100
    return "train" if h < 62 else "val" if h < 70 else "cal" if h < 85 else "test"
