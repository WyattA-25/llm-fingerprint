"""RQ3: Do LLM fingerprints generalize across task domains?

For each target domain D we evaluate on the SAME held-out D test rows with two
training sources, matched in training-set size:
  in-domain:    trained on other prompts from D
  cross-domain: trained only on prompts from the other domains
The gap (in - cross) is the cost of domain shift. Size matching matters: without it,
a drop could just mean "less training data", not "domain shift".

    python experiments/rq3.py [--seeds 0] [--quick]
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from common import cfg_for, fig_path, load, out_path, parse_args, strip_preds, summarize
from src import plots
from src.train import save_json, train_neural, train_tfidf


def group_split(d, frac, seed):
    a, b = next(GroupShuffleSplit(1, test_size=frac, random_state=seed).split(d, groups=d.prompt_id))
    return d.iloc[a], d.iloc[b]


def balanced_sample(d, n, seed):
    """Sample n rows keeping families as balanced as the pool allows."""
    per = n // d.family.nunique()
    return pd.concat([g.sample(min(len(g), per), random_state=seed) for _, g in d.groupby("family")])


def main():
    args = parse_args(lambda ap: ap.add_argument("--domains", nargs="+",
                                                 default=["code", "general", "creative_writing", "math"]))
    df, labels = load(args)
    res = {"labels": labels, "chance": 1 / len(labels), "domains": {}}

    for D in args.domains:
        dom, rest = df[df.domain == D], df[df.domain != D]
        if dom.family.nunique() < len(labels) or len(dom) < 200:
            print(f"skip {D}: too small ({len(dom)})")
            continue
        seed = args.seeds[0]
        dom_trval, dom_test = group_split(dom, 0.30, seed)
        n_train = len(dom_trval)
        in_tr, in_va = group_split(dom_trval, 0.15, seed)
        cross_pool = balanced_sample(rest, n_train, seed)       # size-matched
        cr_tr, cr_va = group_split(cross_pool, 0.15, seed)
        entry = {"n_train": int(n_train), "n_test": int(len(dom_test)), "results": {}}
        print(f"\n== {D}: train={n_train} test={len(dom_test)}")

        for cond, (a, b) in {"in_domain": (in_tr, in_va), "cross_domain": (cr_tr, cr_va)}.items():
            r = train_tfidf(pd.concat([a, b]), {"test": dom_test}, "output", labels)
            entry["results"][f"tfidf/{cond}"] = {"acc": summarize([r]), "f1": summarize([r], "macro_f1"),
                                                 "recall": r["test"]["test"]["recall_per_class"]}
            for m in args.models:
                runs = [train_neural(cfg_for(m, args.quick), a, b, {"test": dom_test}, "output", s, labels,
                                     log=lambda *_: None) for s in args.seeds]
                rec = {l: float(np.mean([x["test"]["test"]["recall_per_class"][l] for x in runs])) for l in labels}
                entry["results"][f"{m}/{cond}"] = {"acc": summarize(runs), "f1": summarize(runs, "macro_f1"),
                                                   "recall": rec}
            print(cond, {k: round(v["acc"]["mean"], 4) for k, v in entry["results"].items() if k.endswith(cond)})
        res["domains"][D] = entry

    models = ["tfidf"] + args.models
    doms = list(res["domains"])
    for m in models:
        series = {c: [res["domains"][d]["results"][f"{m}/{c}"]["acc"]["mean"] for d in doms]
                  for c in ("in_domain", "cross_domain")}
        errs = {c: [res["domains"][d]["results"][f"{m}/{c}"]["acc"]["std"] for d in doms]
                for c in ("in_domain", "cross_domain")}
        plots.grouped_bars(doms, series, f"RQ3 {m.upper()}: in-domain vs cross-domain (test domain on x-axis)",
                           "Accuracy", fig_path(f"rq3_transfer_{m}", args), errs, 1 / len(labels), "chance")

    # Per-family recall drop (in - cross), averaged over models: which fingerprints travel?
    drop = [[np.mean([res["domains"][d]["results"][f"{m}/in_domain"]["recall"][l] -
                      res["domains"][d]["results"][f"{m}/cross_domain"]["recall"][l] for m in models])
             for d in doms] for l in labels]
    res["recall_drop_family_x_domain"] = {l: dict(zip(doms, map(float, row))) for l, row in zip(labels, drop)}
    plots.heatmap(drop, labels, doms, "RQ3 recall drop under domain shift (in - cross)",
                  fig_path("rq3_recall_drop", args), fmt="{:.2f}")
    save_json(res, out_path("rq3", args))


if __name__ == "__main__":
    main()
