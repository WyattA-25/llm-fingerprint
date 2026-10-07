"""RQ4: What characteristics distinguish the LLM families?

1. Descriptive: stylometric features per family (length, lexical diversity,
   markdown, punctuation habits, openers/closers), self-ID and refusal rates.
2. Structure-only model: how far do ~27 hand-crafted features get, and which
   matter most (permutation importance)?
3. What the classifiers use: top TF-IDF n-grams per family, and BiLSTM attention.
4. Ablations: remove one kind of signal, then
     (a) retrain + test on ablated text  -> is the information needed at all?
     (b) test a model trained on clean text on ablated text -> does it rely on it?
   Plus a "no masking" run to measure how much self-identification would inflate results.

    python experiments/rq4.py [--seeds 0] [--quick]
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from common import cfg_for, fig_path, load, out_path, parse_args, splits, summarize
from src import plots
from src.text import ABLATIONS, encode_fields, stylometric_frame
from src.train import _predict, get_device, metrics, save_json, train_neural, train_stylometric, train_tfidf


def ablate(d, name, raw=False):
    d = d.copy()
    if raw:
        d["output"], d["input"] = d.output_raw, d.input_raw
    d["output"] = d.output.map(ABLATIONS[name])
    return d


def lstm_attention_tokens(model, vocab, te, labels, cfg, k=15, min_count=30):
    device = get_device()
    seqs = encode_fields(vocab, te.input.tolist(), te.output.tolist(), "output", cfg.max_len)
    fam = te.family.values
    s, c = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(int))
    model.eval()
    import torch
    for i in range(0, len(seqs), 64):
        chunk = seqs[i:i + 64]
        lens = torch.tensor([len(x) for x in chunk])
        x = torch.zeros(len(chunk), int(lens.max()), dtype=torch.long)
        for r, q in enumerate(chunk):
            x[r, :len(q)] = torch.tensor(q)
        with torch.no_grad():
            model(x.to(device), lens.to(device))
        w = model.last_attention.cpu().numpy()
        for r, q in enumerate(chunk):
            f = fam[i + r]
            for pos, tok in enumerate(q):
                t = vocab.itos[tok]
                s[f][t] += w[r, pos] * len(q)   # scale by length: 1.0 = uniform attention
                c[f][t] += 1
    out = {}
    for f in labels:
        cand = [(t, s[f][t] / c[f][t]) for t in c[f] if c[f][t] >= min_count]
        out[f] = [(t, round(float(v), 2)) for t, v in sorted(cand, key=lambda z: -z[1])[:k]]
    return out


def main():
    args = parse_args(lambda ap: ap.add_argument("--ablation_models", nargs="+", default=["tfidf", "cnn"]))
    df, labels = load(args)
    tr, va, te = splits(df)
    res = {"labels": labels, "chance": 1 / len(labels)}

    # 1. Descriptive statistics
    F = stylometric_frame(df.output)
    F["family"] = df.family.values
    means = F.groupby("family").mean().loc[labels]
    sd = F.drop(columns="family").std().replace(0, np.nan)
    z = ((means - F.drop(columns="family").mean()) / sd).fillna(0.0)
    res["feature_means"] = means.round(4).to_dict("index")
    plots.heatmap(z.T.values, list(z.columns), labels, "RQ4 stylometric profile (z-score vs. all responses)",
                  fig_path("rq4_style_heatmap", args), fmt="{:.1f}")
    res["flags_by_family"] = df.groupby("family")[["self_id", "refusal", "is_reasoning"]].mean().round(4).to_dict("index")
    res["length_words_median"] = df.assign(w=df.output.str.split().str.len()).groupby("family").w.median().to_dict()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.boxplot([np.log10(df[df.family == f].output.str.split().str.len().clip(lower=1)) for f in labels],
               showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("log10(words)")
    ax.set_title("Response length by family")
    fig.tight_layout()
    fig.savefig(fig_path("rq4_length_box", args), dpi=180)
    plt.close(fig)

    # 2. Structure-only model + permutation importance
    sty, clf, _, frames = train_stylometric(tr, {"test": te}, labels, return_model=True)
    res["stylometric_model"] = {"accuracy": sty["test"]["test"]["accuracy"], "macro_f1": sty["test"]["test"]["macro_f1"]}
    pi = permutation_importance(clf, frames["test"], te.family.values, n_repeats=5, random_state=0, scoring="accuracy")
    imp = pd.Series(pi.importances_mean, index=frames["test"].columns).sort_values(ascending=False)
    res["stylometric_importance"] = imp.round(4).to_dict()
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.head(15)[::-1].plot.barh(ax=ax)
    ax.set_xlabel("Accuracy drop when feature is shuffled")
    ax.set_title("RQ4 most important structural features")
    fig.tight_layout()
    fig.savefig(fig_path("rq4_feature_importance", args), dpi=180)
    plt.close(fig)
    print("stylometric-only acc:", round(sty["test"]["test"]["accuracy"], 4))

    # 3a. Top TF-IDF n-grams per family
    _, lr, vecs = train_tfidf(tr, {"test": te}, "output", labels, return_model=True)
    names = np.concatenate([v.get_feature_names_out() for v in vecs])
    res["tfidf_top_ngrams"] = {l: [repr(str(names[j])) for j in np.argsort(lr.coef_[i])[::-1][:20]]
                               for i, l in enumerate(lr.classes_)}

    # 3b. BiLSTM attention
    if "lstm" in args.models:
        cfg = cfg_for("lstm", args.quick)
        _, lstm, vocab = train_neural(cfg, tr, va, {"test": te}, "output", args.seeds[0], labels, return_model=True)
        res["lstm_attention_top_tokens"] = lstm_attention_tokens(lstm, vocab, te, labels, cfg)

    # 4. Ablations
    abl = {}
    variants = list(ABLATIONS) + ["no_masking"]
    for m in args.ablation_models:
        abl[m] = {}
        clean_model = None
        for v in variants:
            name, raw = ("none", True) if v == "no_masking" else (v, False)
            a_tr, a_va, a_te = (ablate(x, name, raw) for x in (tr, va, te))
            if m == "tfidf":
                r = train_tfidf(a_tr, {"test": a_te}, "output", labels)
                abl[m][v] = {"retrain": summarize([r])}
            else:
                runs = []
                for s in args.seeds:
                    out = train_neural(cfg_for(m, args.quick), a_tr, a_va, {"test": a_te}, "output", s, labels,
                                       log=lambda *_: None, return_model=(v == "none" and s == args.seeds[0]))
                    if isinstance(out, tuple):
                        out, clean_model, clean_vocab = out
                    runs.append(out)
                abl[m][v] = {"retrain": summarize(runs)}
            print(f"ablation {m}/{v}: retrain acc={abl[m][v]['retrain']['mean']:.4f}")
        if clean_model is not None:   # test-time ablation with the clean-trained model
            cfg = cfg_for(m, args.quick)
            for v in variants:
                name, raw = ("none", True) if v == "no_masking" else (v, False)
                a_te = ablate(te, name, raw)
                seqs = encode_fields(clean_vocab, a_te.input.tolist(), a_te.output.tolist(), "output", cfg.max_len)
                p, _ = _predict(clean_model, seqs, cfg.batch_size, get_device())
                abl[m][v]["test_time"] = metrics(a_te.family.values, [labels[i] for i in p], labels)["accuracy"]
    res["ablations"] = abl

    order = [v for v in variants if v != "none"]
    for m in abl:
        base = abl[m]["none"]["retrain"]["mean"]
        series = {"retrain on ablated text": [abl[m][v]["retrain"]["mean"] for v in order]}
        if "test_time" in abl[m]["none"]:
            series["clean model, ablated test"] = [abl[m][v]["test_time"] for v in order]
        plots.grouped_bars(order, series, f"RQ4 ablations, {m.upper()} (dashed = no ablation, {base:.3f})",
                           "Accuracy", fig_path(f"rq4_ablation_{m}", args), hline=base, hline_label="no ablation")

    save_json(res, out_path("rq4", args))


if __name__ == "__main__":
    main()
