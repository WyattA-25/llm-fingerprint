"""Run the whole pipeline in order (cross-platform).

    python run_all.py                # full run on data/dataset.parquet
    python run_all.py --smoke        # synthetic data, tiny settings, ~2 min on CPU
"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
ap.add_argument("--only", nargs="+", choices=["preprocess", "rq1", "rq2", "rq3", "rq4"],
                help="run only these steps (finished training runs are cached either way)")
args = ap.parse_args()

py = sys.executable
if args.smoke:
    raw, proc, extra, seeds = "data/mock_dataset.parquet", "data/mock_processed.parquet", ["--quick", "--tag", "mock"], ["0"]
    steps = [[py, "scripts/make_mock_data.py"]]
else:
    raw, proc, extra, seeds = "data/dataset.parquet", "data/processed.parquet", [], args.seeds
    steps = []

common = ["--data", proc, *extra]
steps += [
    [py, "-m", "src.preprocess", "--in", raw, "--out", proc],
    [py, "experiments/rq1.py", *common, "--seeds", *seeds],
    [py, "experiments/rq2.py", *common, "--seeds", *seeds],
    [py, "experiments/rq3.py", *common, "--seeds", *seeds] + (["--domains", "code"] if args.smoke else []),
    [py, "experiments/rq4.py", *common, "--seeds", *seeds],
]
if args.only:
    steps = [s for s in steps if any(k in " ".join(s) for k in args.only) or "make_mock" in " ".join(s)]
for s in steps:
    print("\n>>>", " ".join(s), flush=True)
    subprocess.run(s, check=True)
