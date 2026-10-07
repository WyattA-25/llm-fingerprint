"""RQ2: Does the user's prompt help identify the LLM?

Compares input-only, output-only, and input+output for TF-IDF, CNN, and BiLSTM.
Then tests *why* input-only might work, with two controls:
  1. Matched pairs: prompts answered by two different families. Input is identical,
     so an input-only model must give both rows the same label (accuracy <= 50%).
  2. Timestamp-only model: if a classifier can predict the family from *when* the
     prompt was asked, prompts differ across families because of the Arena's
     model rotation over time, not because of anything about the models.

    python experiments/rq2.py [--seeds 0 1 2] [--quick]
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from common import cfg_for, fig_path, load, out_path, parse_args, splits, strip_preds, summarize
from src import plots
from src.train import metrics, save_json, train_neural, train_tfidf

MODES = ["input", "output", "both"]


def time_features(d):
    t = pd.to_datetime(d.timestamp)
    return pd.DataFrame({"day": (t - pd.Timestamp("2025-01-01")).dt.days, "hour": t.dt.hour})


def top_input_features(clf, vecs, labels, k=12):
    """Most predictive word/char n-grams per family for the input-only TF-IDF model."""
    names = np.concatenate([v.get_feature_names_out() for v in vecs])
    out = {}
    for i, l in enumerate(clf.classes_):
        idx = np.argsort(clf.coef_[i])[::-1][:k]
        out[l] = [str(names[j]) for j in idx]
    return out


def main():
    args = parse_args(lambda ap: ap.add_argument("--neural_modes", nargs="+", default=MODES,
                                                 help="restrict neural runs, e.g. to skip re-training output-only"))
    df, labels = load(args)
    tr, va, te = splits(df)
    te_matched = te[te.matched_pair]
    tests = {"test": te, "test_matched": te_matched}
    print(f"test={len(te)} matched-pair test rows={len(te_matched)}")
    res = {"labels": labels, "chance": 1 / len(labels), "summary": {}, "runs": {}}

    for mode in MODES:
        r, clf, vecs = train_tfidf(tr, tests, mode, labels, return_model=True)
        res["runs"][f"tfidf/{mode}"] = [strip_preds(r)]
        res["summary"][f"tfidf/{mode}"] = {t: summarize([r], "accuracy", t) for t in tests}
        if mode == "input":
            res["input_only_top_features"] = top_input_features(clf, vecs, labels)
            # Same prediction for both rows of a pair => pair-level check
            p = pd.Series(r["test"]["test_matched"]["pred"], index=te_matched.index)
            same = te_matched.assign(pred=p).groupby("prompt_id").pred.nunique().eq(1).mean()
            res["input_only_same_pred_within_pair"] = float(same)
        print(f"tfidf/{mode}", {t: round(r["test"][t]["accuracy"], 4) for t in tests})

    for m in args.models:
        for mode in args.neural_modes:
            runs = [train_neural(cfg_for(m, args.quick), tr, va, tests, mode, s, labels) for s in args.seeds]
            res["runs"][f"{m}/{mode}"] = [strip_preds(x) for x in runs]
            res["summary"][f"{m}/{mode}"] = {t: summarize(runs, "accuracy", t) for t in tests}
            print(f"{m}/{mode}", {t: round(res["summary"][f"{m}/{mode}"][t]["mean"], 4) for t in tests})

    # Control 2: timestamp-only classifier
    gbm = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(time_features(tr), tr.family)
    res["timestamp_only"] = {t: metrics(d.family.values, gbm.predict(time_features(d)), labels) for t, d in tests.items()}
    res["timestamp_range_by_family"] = (df.assign(t=pd.to_datetime(df.timestamp)).groupby("family").t
                                        .agg(["min", "median", "max"]).astype(str).to_dict("index"))
    print("timestamp-only acc:", round(res["timestamp_only"]["test"]["accuracy"], 4))

    models = ["tfidf"] + [m for m in args.models if all(f"{m}/{x}" in res["summary"] for x in MODES)]
    for t, title in (("test", "full test set"), ("test_matched", "matched-prompt pairs")):
        series = {mode: [res["summary"][f"{m}/{mode}"][t]["mean"] for m in models] for mode in MODES}
        errs = {mode: [res["summary"][f"{m}/{mode}"][t]["std"] for m in models] for mode in MODES}
        plots.grouped_bars([m.upper() for m in models], series, f"RQ2 accuracy by input field ({title})",
                           "Accuracy", fig_path(f"rq2_fields_{t}", args), errs, 1 / len(labels), "chance")

    # Family timeline: when each family appears in the data
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, f in enumerate(labels):
        t = pd.to_datetime(df[df.family == f].timestamp)
        ax.violinplot([t.map(pd.Timestamp.toordinal)], [i], vert=False, showextrema=False)
    ax.set_yticks(range(len(labels)), labels)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks, [pd.Timestamp.fromordinal(int(x)).strftime("%b %d") for x in ticks], fontsize=8)
    ax.set_title("When each family's responses were collected")
    fig.tight_layout()
    fig.savefig(fig_path("rq2_timeline", args), dpi=180)
    plt.close(fig)

    save_json(res, out_path("rq2", args))


if __name__ == "__main__":
    main()
