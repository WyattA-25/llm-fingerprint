"""Training and evaluation for the CNN, BiLSTM, and non-neural baselines."""
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from .models import build_model
from .text import Vocab, encode_fields, field_text, stylometric_frame

FAMILIES = ["Claude", "DeepSeek", "GPT", "Gemini", "Grok", "Llama", "Mistral", "Qwen"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    model: str = "cnn"               # cnn | lstm
    level: str = "char"              # char | word
    max_len: int = 2000
    vocab_size: int = 300
    min_freq: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    patience: int = 4
    model_kw: dict = field(default_factory=dict)


DEFAULTS = {
    "cnn": Config(model="cnn", level="char", max_len=2000, vocab_size=300, min_freq=5),
    "lstm": Config(model="lstm", level="word", max_len=400, vocab_size=30000, min_freq=3),
}


def metrics(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_per_class": dict(zip(labels, recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).round(4).tolist())),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------- neural models
def _batches(seqs, labels, bs, shuffle, rng):
    """Length-bucketed batches: sort within chunks so padding is small, then shuffle batches."""
    idx = np.arange(len(seqs))
    if shuffle:
        rng.shuffle(idx)
    chunk = bs * 50
    batches = []
    for i in range(0, len(idx), chunk):
        c = sorted(idx[i:i + chunk], key=lambda j: len(seqs[j]))
        batches += [c[k:k + bs] for k in range(0, len(c), bs)]
    if shuffle:
        rng.shuffle(batches)
    for b in batches:
        lens = torch.tensor([len(seqs[j]) for j in b])
        x = torch.zeros(len(b), int(lens.max()), dtype=torch.long)
        for r, j in enumerate(b):
            x[r, :lens[r]] = torch.tensor(seqs[j])
        yield b, x, lens, torch.tensor([labels[j] for j in b])


@torch.no_grad()
def _predict(model, seqs, bs, device):
    model.eval()
    dummy = [0] * len(seqs)
    preds = np.zeros(len(seqs), dtype=int)
    probs = np.zeros((len(seqs), model.fc.out_features), dtype=np.float32)
    for b, x, lens, _ in _batches(seqs, dummy, bs * 2, False, None):
        logits = model(x.to(device), lens.to(device))
        p = torch.softmax(logits, 1).cpu().numpy()
        probs[b], preds[b] = p, p.argmax(1)
    return preds, probs


# ---------------------------------------------------------------- run cache (resume after crashes)
CACHE_DIR = Path(os.environ.get("FP_CACHE", Path(__file__).resolve().parents[1] / "results" / "cache"))


def _df_hash(d):
    h = hashlib.md5()
    for x, y, f in zip(d.input, d.output, d.family):
        h.update(f"{f}\x1f{x}\x1f{y}\x1e".encode())
    return h.hexdigest()


def _run_key(kind, params, train_df, val_df, test_sets):
    blob = json.dumps({"kind": kind, "params": params,
                       "train": _df_hash(train_df), "val": _df_hash(val_df) if val_df is not None else None,
                       "test": {k: _df_hash(v) for k, v in sorted(test_sets.items())}}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:20]


def train_neural(cfg: Config, train_df, val_df, test_sets: dict, mode="output", seed=0,
                 labels=FAMILIES, log=print, return_model=False):
    """Cached wrapper: identical (config, data, seed) runs are loaded from results/cache
    instead of retrained, so a crashed pipeline resumes where it stopped."""
    key = _run_key("neural", {"cfg": asdict(cfg), "mode": mode, "seed": seed, "labels": labels},
                   train_df, val_df, test_sets)
    jpath, mpath = CACHE_DIR / f"{key}.json", CACHE_DIR / f"{key}.pt"
    if jpath.exists() and (not return_model or mpath.exists()):
        out = json.loads(jpath.read_text())
        log(f"  [{cfg.model}/{mode}/seed{seed}] cached")
        if not return_model:
            return out
        ck = torch.load(mpath, map_location="cpu")
        vocab = Vocab.from_itos(ck["itos"], cfg.level)
        model = build_model(cfg.model, len(vocab), len(labels), **cfg.model_kw)
        model.load_state_dict(ck["state"])
        return out, model.to(get_device()), vocab
    out, model, vocab = _train_neural(cfg, train_df, val_df, test_sets, mode, seed, labels, log)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if return_model:
        torch.save({"state": model.state_dict(), "itos": vocab.itos}, mpath)
    save_json(out, jpath)
    return (out, model, vocab) if return_model else out


def _train_neural(cfg: Config, train_df, val_df, test_sets: dict, mode="output", seed=0,
                  labels=FAMILIES, log=print):
    """Train on train_df, early-stop on val macro-F1, evaluate on each test set.
    test_sets: {name: DataFrame}. Returns results dict (and optionally model+vocab)."""
    set_seed(seed)
    device = get_device()
    lab2id = {l: i for i, l in enumerate(labels)}

    fit_texts = field_text(train_df, "both" if mode == "both" else mode)
    vocab = Vocab(fit_texts, cfg.level, cfg.vocab_size, cfg.min_freq)
    enc = lambda d: encode_fields(vocab, d.input.tolist(), d.output.tolist(), mode, cfg.max_len)
    Xtr, Xva = enc(train_df), enc(val_df)
    ytr = train_df.family.map(lab2id).tolist()
    yva = val_df.family.map(lab2id).values

    model = build_model(cfg.model, len(vocab), len(labels), **cfg.model_kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(cfg.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for _, x, lens, y in _batches(Xtr, ytr, cfg.batch_size, True, rng):
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, lens), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        vp, _ = _predict(model, Xva, cfg.batch_size, device)
        vf1 = f1_score(yva, vp, average="macro")
        sched.step(vf1)
        history.append({"epoch": ep + 1, "train_loss": tot / n, "val_macro_f1": float(vf1), "sec": time.time() - t0})
        log(f"  [{cfg.model}/{mode}/seed{seed}] ep{ep + 1} loss={tot / n:.3f} val_f1={vf1:.3f} ({time.time() - t0:.0f}s)")
        if vf1 > best:
            best, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)

    out = {"config": asdict(cfg), "mode": mode, "seed": seed, "vocab_len": len(vocab),
           "n_params": sum(p.numel() for p in model.parameters()),
           "best_val_macro_f1": float(best), "history": history, "test": {}}
    for name, d in test_sets.items():
        p, _ = _predict(model, enc(d), cfg.batch_size, device)
        out["test"][name] = metrics(d.family.values, [labels[i] for i in p], labels)
        out["test"][name]["pred"] = [labels[i] for i in p]
    return out, model, vocab


# ---------------------------------------------------------------- baselines
def tfidf_features(train_texts, other_texts):
    w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=50000, min_df=2,
                        sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    c = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=100000, min_df=3,
                        sublinear_tf=True, lowercase=False)
    Xtr = hstack([w.fit_transform(train_texts), c.fit_transform(train_texts)]).tocsr()
    Xo = [hstack([w.transform(t), c.transform(t)]).tocsr() for t in other_texts]
    return Xtr, Xo, (w, c)


def train_tfidf(train_df, test_sets: dict, mode="output", labels=FAMILIES, return_model=False, C=4.0):
    if not return_model:
        key = _run_key("tfidf", {"mode": mode, "C": C, "labels": labels}, train_df, None, test_sets)
        jpath = CACHE_DIR / f"{key}.json"
        if jpath.exists():
            return json.loads(jpath.read_text())
        out = _train_tfidf(train_df, test_sets, mode, labels, False, C)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        save_json(out, jpath)
        return out
    return _train_tfidf(train_df, test_sets, mode, labels, True, C)


def _train_tfidf(train_df, test_sets: dict, mode="output", labels=FAMILIES, return_model=False, C=4.0):
    names = list(test_sets)
    Xtr, Xte, vecs = tfidf_features(field_text(train_df, mode), [field_text(test_sets[n], mode) for n in names])
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(Xtr, train_df.family.values)
    out = {"model": "tfidf_logreg", "mode": mode, "test": {}}
    for n, X in zip(names, Xte):
        p = clf.predict(X)
        out["test"][n] = metrics(test_sets[n].family.values, p, labels)
        out["test"][n]["pred"] = p.tolist()
    return (out, clf, vecs) if return_model else out


def train_stylometric(train_df, test_sets: dict, labels=FAMILIES, return_model=False, seed=0):
    """Hand-crafted structural/linguistic features only (no vocabulary)."""
    Ftr = stylometric_frame(train_df.output)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=seed)
    clf.fit(Ftr, train_df.family.values)
    out = {"model": "stylometric_gbm", "features": list(Ftr.columns), "test": {}}
    frames = {}
    for n, d in test_sets.items():
        F = stylometric_frame(d.output)
        frames[n] = F
        p = clf.predict(F)
        out["test"][n] = metrics(d.family.values, p, labels)
    return (out, clf, Ftr, frames) if return_model else out


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
