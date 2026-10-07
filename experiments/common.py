"""Shared helpers for the RQ scripts."""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import DEFAULTS, FAMILIES  # noqa: E402

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def parse_args(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/processed.parquet"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=["cnn", "lstm"])
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--tag", default="", help="suffix for result files (e.g. 'mock')")
    if extra:
        extra(ap)
    return ap.parse_args()


def load(args):
    df = pd.read_parquet(args.data)
    labels = [f for f in FAMILIES if f in set(df.family)]
    return df, labels


def splits(df):
    return df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]


def cfg_for(name, quick=False):
    c = copy.deepcopy(DEFAULTS[name])
    if quick:
        c.epochs, c.patience = 2, 1
        c.max_len = min(c.max_len, 300)
    return c


def summarize(runs, key="accuracy", test="test"):
    vals = [r["test"][test][key] for r in runs]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "runs": vals}


def out_path(name, args):
    return RESULTS / (f"{name}_{args.tag}.json" if args.tag else f"{name}.json")


def fig_path(name, args):
    return FIGS / (f"{name}_{args.tag}.png" if args.tag else f"{name}.png")


def strip_preds(res):
    """Drop per-row predictions before saving summaries (keeps JSON small)."""
    r = copy.deepcopy(res)
    for t in r.get("test", {}).values():
        t.pop("pred", None)
    return r
