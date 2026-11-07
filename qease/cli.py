"""Command line interface:  python -m qease --stages pool load probe train eval crc analyze --set MODEL_ID=Qwen/Qwen2-VL-2B-Instruct"""
import argparse
import ast

from qease.pipeline import STAGE_ORDER, run


def _parse_value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qease", description="Q-EASE: calibrated evidence sufficiency for efficient VLMs")
    ap.add_argument("--stages", nargs="+", choices=STAGE_ORDER, help="stages to run (default: the STAGES of the configuration)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override configuration keys, e.g. --set MODEL_ID=Qwen/Qwen2-VL-2B-Instruct PROBE_TOTAL=3000")
    ap.add_argument("--base-dir", default=None, help="output folder (default: /kaggle/working/qease on Kaggle, ./qease_out elsewhere)")
    args = ap.parse_args(argv)
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = _parse_value(value)
    run(stages=args.stages, overrides=overrides, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
