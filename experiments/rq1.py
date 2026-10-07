"""RQ1: Can we identify which LLM family generated a response (output only)?

Trains the CNN (char) and BiLSTM (word) over several seeds, plus two baselines:
TF-IDF + logistic regression and a stylometric-features-only model.
Also breaks accuracy down by domain, reasoning variant, and refusals.

    python experiments/rq1.py [--seeds 0 1 2] [--quick]
"""
import numpy as np
import pandas as pd

from common import cfg_for, fig_path, load, out_path, parse_args, splits, strip_preds, summarize
from src import plots
from src.train import save_json, train_neural, train_stylometric, train_tfidf


def breakdown(test_df, preds):
    d = test_df.assign(correct=np.array(preds) == test_df.family.values)
    return {
        "by_domain": d.groupby("domain").correct.mean().round(4).to_dict(),
        "by_reasoning_variant": d.groupby("is_reasoning").correct.mean().round(4).rename(str).to_dict(),
        "by_refusal": d.groupby("refusal").correct.mean().round(4).rename(str).to_dict(),
        "by_length_quartile": d.groupby(pd.qcut(d.output.str.len(), 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"]),
                                        observed=True).correct.mean().round(4).to_dict(),
        "by_model": d.groupby("model").correct.agg(["mean", "size"]).round(4).to_dict("index"),
    }


def main():
    args = parse_args()
    df, labels = load(args)
    tr, va, te = splits(df)
    print(f"train={len(tr)} val={len(va)} test={len(te)} classes={labels}")
    results = {"labels": labels, "chance": 1 / len(labels), "summary": {}, "runs": {}, "breakdown": {}}

    base = train_tfidf(tr, {"test": te}, "output", labels)
    results["runs"]["tfidf"] = [strip_preds(base)]
    results["summary"]["tfidf"] = {k: summarize([base], k) for k in ("accuracy", "macro_f1")}
    results["breakdown"]["tfidf"] = breakdown(te, base["test"]["test"]["pred"])
    print("TF-IDF acc:", round(base["test"]["test"]["accuracy"], 4))

    sty = train_stylometric(tr, {"test": te}, labels)
    results["runs"]["stylometric"] = [sty]
    results["summary"]["stylometric"] = {k: summarize([sty], k) for k in ("accuracy", "macro_f1")}
    print("Stylometric acc:", round(sty["test"]["test"]["accuracy"], 4))

    for m in args.models:
        runs = [train_neural(cfg_for(m, args.quick), tr, va, {"test": te}, "output", s, labels) for s in args.seeds]
        results["summary"][m] = {k: summarize(runs, k) for k in ("accuracy", "macro_f1")}
        best = max(runs, key=lambda r: r["test"]["test"]["macro_f1"])
        results["breakdown"][m] = breakdown(te, best["test"]["test"]["pred"])
        results["runs"][m] = [strip_preds(r) for r in runs]
        plots.confusion(best["test"]["test"]["confusion"], labels,
                        f"RQ1 {m.upper()} (output only), acc={best['test']['test']['accuracy']:.3f}",
                        fig_path(f"rq1_confusion_{m}", args))
        print(m, results["summary"][m]["accuracy"])

    plots.confusion(base["test"]["test"]["confusion"], labels,
                    f"RQ1 TF-IDF+LR, acc={base['test']['test']['accuracy']:.3f}", fig_path("rq1_confusion_tfidf", args))
    plots.curves({m: results["runs"][m][0]["history"] for m in args.models}, fig_path("rq1_training_curves", args))

    names = ["tfidf", "stylometric"] + args.models
    rec = {n: [np.mean([r["test"]["test"]["recall_per_class"][l] for r in results["runs"][n]]) for l in labels]
           for n in names}
    plots.grouped_bars(labels, rec, "RQ1 per-family recall (output only)", "Recall",
                       fig_path("rq1_recall_per_family", args), hline=1 / len(labels), hline_label="chance")

    save_json(results, out_path("rq1", args))
    print("\nSummary (accuracy):")
    for n in names:
        s = results["summary"][n]["accuracy"]
        print(f"  {n:12s} {s['mean']:.4f} ± {s['std']:.4f}")


if __name__ == "__main__":
    main()
