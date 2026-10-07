"""Figure helpers (matplotlib only, so it runs anywhere)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def confusion(cm, labels, title, path):
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def grouped_bars(groups, series: dict, title, ylabel, path, errs: dict = None, hline=None, hline_label=None):
    """groups: x tick labels; series: {name: [values per group]}."""
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(groups) + 2), 4))
    for k, (name, vals) in enumerate(series.items()):
        e = errs.get(name) if errs else None
        ax.bar(x + k * w - 0.4 + w / 2, vals, w, yerr=e, capsize=3, label=name)
    if hline is not None:
        ax.axhline(hline, ls="--", c="gray", lw=1, label=hline_label)
    ax.set_xticks(x, groups, rotation=20 if len(groups) > 5 else 0, ha="right" if len(groups) > 5 else "center")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def curves(histories: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], marker="o", label=name)
        axes[1].plot(ep, [r["val_macro_f1"] for r in h], marker="o", label=name)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation macro-F1")
    for a in axes:
        a.set_xlabel("Epoch")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def heatmap(matrix, rows, cols, title, path, cmap="RdBu_r", center=0.0, fmt="{:.1f}"):
    m = np.asarray(matrix, dtype=float)
    lim = np.nanmax(np.abs(m - center)) or 1
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(cols), 0.9 + 0.32 * len(rows)))
    im = ax.imshow(m, cmap=cmap, vmin=center - lim, vmax=center + lim, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
