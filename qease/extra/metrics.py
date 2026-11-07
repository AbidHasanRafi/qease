"""Official answer metrics: VQA accuracy (leave-one-out), ANLS (tau=0.5) and exact match, with the standard VQA answer normalisation."""
import re

import numpy as np


_NUMW = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


_ARTS = {"a", "an", "the"}


_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


_PERIOD = re.compile(r"(?!<=\d)(\.)(?!\d)")


_COMMA = re.compile(r"(\d)(,)(\d)")


def vqa_norm(s):
    """official VQA answer normalisation (punctuation, digits, articles)."""
    s = str(s).replace("\n", " ").replace("\t", " ").strip().lower()
    out = s
    for p in _PUNCT:
        if (p + " " in s or " " + p in s) or (re.search(_COMMA, s) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD.sub("", out)
    return " ".join(_NUMW.get(w, w) for w in out.split() if w not in _ARTS)


def vqa_acc(pred, answers):
    """official VQA accuracy: mean over leave-one-out subsets of min(1, #matches/3)."""
    p = vqa_norm(pred); g = [vqa_norm(a) for a in answers]
    if not g: return 0.0
    if len(g) < 2: return float(p == g[0])
    return float(np.mean([min(1.0, sum(o == p for o in (g[:i] + g[i + 1:])) / 3.0) for i in range(len(g))]))


def x_lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred, answers, tau=0.5):
    p = " ".join(str(pred).lower().split()); best = 0.0
    for a in answers:
        g = " ".join(str(a).lower().split())
        d = x_lev(p, g) / max(1, max(len(p), len(g)))
        best = max(best, 1 - d if d < tau else 0.0)
    return best


def x_score(ds, pred, answers):
    ds = ds[4:] if ds.startswith("std_") else ds                                          # standard benchmarks use the same metric as their pool twin
    if ds == "pope":                                                                       # POPE: yes/no accuracy on the first word of the answer
        w = vqa_norm(pred).split(); return float(bool(w) and w[0] == vqa_norm(answers[0]))
    if ds in ("vqav2", "textvqa", "aokvqa"): return vqa_acc(pred, answers)
    if ds == "docvqa": return anls(pred, answers)
    return float(vqa_norm(pred) == vqa_norm(answers[0])) if answers else 0.0          # GQA, VSR, CLEVR: exact match
