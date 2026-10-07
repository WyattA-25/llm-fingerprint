"""Regenerate report figures from results/*.json (no retraining)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import plots  # noqa: E402

R = ROOT / "results"
F = R / "figures"
r3 = json.loads((R / "rq3.json").read_text())
labels, doms = r3["labels"], list(r3["domains"])
models = ["tfidf", "cnn", "lstm"]

# Recall-drop heatmap with a readable size/title
drop = [[r3["recall_drop_family_x_domain"][l][d] for d in doms] for l in labels]
plots.heatmap(drop, labels, doms, "Recall drop under domain shift (in-domain minus cross-domain)",
              F / "rq3_recall_drop.png", fmt="{:.2f}")

# One compact figure: accuracy drop (in - cross) per model and target domain
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

x = np.arange(len(doms))
w = 0.26
fig, ax = plt.subplots(figsize=(7, 3.4))
for k, m in enumerate(models):
    vals, errs = [], []
    for d in doms:
        a = r3["domains"][d]["results"][f"{m}/in_domain"]["acc"]
        b = r3["domains"][d]["results"][f"{m}/cross_domain"]["acc"]
        vals.append(a["mean"] - b["mean"])
        errs.append(np.sqrt(a["std"] ** 2 + b["std"] ** 2))
    ax.bar(x + (k - 1) * w, vals, w, yerr=errs, capsize=3, label=m.upper())
ax.axhline(0, c="black", lw=0.8)
ax.set_xticks(x, [f"{d}\n(n_test={r3['domains'][d]['n_test']})" for d in doms], fontsize=8)
ax.set_ylabel("Accuracy drop (in - cross)")
ax.set_title("RQ3: cost of testing on an unseen domain (size-matched training)", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(F / "rq3_drop_summary.png", dpi=180)
plt.close(fig)
print("figures regenerated")
