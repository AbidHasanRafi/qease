"""Command line interface of the extra experiments:  python -m qease.extra --set RUN_MODEL2=False BENCH_N_PER_DS=80"""
import argparse
import ast

from qease.extra.pipeline import run_extra


def _parse_value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qease.extra", description="Extra experiments: trained baselines, question swap, accuracy, standard benchmarks, second model")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override keys of qease.extra.config.DEFAULT_XCFG, e.g. --set RUN_MODEL2=False STD_N=80")
    ap.add_argument("--base-dir", default=None, help="folder that holds the first run's outputs (default: /kaggle/working/qease on Kaggle, ./qease_out elsewhere)")
    args = ap.parse_args(argv)
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = _parse_value(value)
    run_extra(overrides, args.base_dir)


if __name__ == "__main__":
    main()
