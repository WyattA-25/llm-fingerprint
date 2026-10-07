"""
Step 1: Build the (LLM_family, LLM_input, LLM_output) dataset from
lmarena-ai/arena-human-preference-140k.

Run:
    pip install datasets pandas pyarrow
    huggingface-cli login          # only if the dataset asks you to accept terms
    python prepare_data.py --inspect   # print schema + model counts, no output file
    python prepare_data.py             # build data/dataset.parquet

Each Arena battle = one prompt answered by two models. We keep both sides as
separate rows sharing a prompt_id, so splits can be grouped by prompt (no leakage)
and we get a built-in "matched prompt" subset for RQ2.
"""
import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Order matters: first match wins.
FAMILY_PATTERNS = [
    ("GPT", r"^(gpt-(?!oss)|chatgpt|o1|o3|o4)"),
    ("Claude", r"claude"),
    ("Gemini", r"gemini"),          # Gemma excluded on purpose (different tier)
    ("Llama", r"llama"),
    ("Qwen", r"qwen|qwq"),
    ("DeepSeek", r"deepseek"),
    ("Mistral", r"mistral|mixtral|magistral"),
    ("Grok", r"grok"),
]


# Reasoning/"thinking" variants: flagged so we can test whether they behave differently.
REASONING = re.compile(r"(?<!no-)thinking|deepseek-r1|qwq|magistral|^o\d|grok-3-mini|qwen3-(235b-a22b|30b-a3b)$", re.I)


def model_to_family(name: str):
    name = (name or "").lower()
    for fam, pat in FAMILY_PATTERNS:
        if re.search(pat, name):
            return fam
    return None


def msg_text(msg) -> str:
    """Content may be a plain string or a list of typed parts."""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type", "text") == "text")
    return ""


def single_turn(conv):
    """Return (prompt, response) if conversation is exactly one user + one assistant turn."""
    if conv is None or len(conv) != 2:
        return None
    u, a = conv[0], conv[1]
    if u.get("role") != "user" or a.get("role") != "assistant":
        return None
    return msg_text(u).strip(), msg_text(a).strip()


def find_flag(tags, key):
    """category_tag is nested (e.g. {'math_v0.1': {'math': True}}). Search recursively."""
    if isinstance(tags, dict):
        for k, v in tags.items():
            if k == key and isinstance(v, bool):
                return v
            r = find_flag(v, key)
            if r is not None:
                return r
    return None


CODE_HINT = re.compile(r"```|\bdef |\bclass |\bfunction\b|#include|\bSELECT\b|\bpython\b|\bjavascript\b|\bcode\b", re.I)


def domain_of(row, prompt):
    tags = row.get("category_tag")
    is_code = row.get("is_code")
    if is_code is None:
        is_code = bool(CODE_HINT.search(prompt))
    if is_code:
        return "code"
    if find_flag(tags, "math"):
        return "math"
    if find_flag(tags, "creative_writing"):
        return "creative_writing"
    return "general"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--per_family", type=int, default=2000, help="cap per family after balancing")
    ap.add_argument("--min_family", type=int, default=800, help="drop families with fewer rows")
    ap.add_argument("--seed", type=int, default=448)
    ap.add_argument("--out", default="data/dataset.parquet")
    args = ap.parse_args()

    ds = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")
    print("Columns:", ds.column_names)
    print("Rows:", len(ds))

    if args.inspect:
        ex = ds[0]
        for k, v in ex.items():
            print(f"--- {k}: {str(v)[:300]}")
        models = pd.Series(list(ds["model_a"]) + list(ds["model_b"])).value_counts()
        print("\nTop models:\n", models.head(60).to_string())
        return

    rows = []
    for i, r in enumerate(ds):
        lang = r.get("language")
        if lang is not None and lang not in ("English", "en"):
            continue
        for side in ("a", "b"):
            fam = model_to_family(r.get(f"model_{side}"))
            if fam is None:
                continue
            pr = single_turn(r.get(f"conversation_{side}"))
            if pr is None:
                continue
            prompt, resp = pr
            if not (10 <= len(prompt) <= 4000 and 50 <= len(resp) <= 8000):
                continue
            rows.append({
                "prompt_id": hashlib.md5(prompt.encode()).hexdigest()[:16],
                "battle_id": r.get("id", i),
                "model": r.get(f"model_{side}"),
                "is_reasoning": bool(REASONING.search(r.get(f"model_{side}") or "")),
                "family": fam,
                "domain": domain_of(r, prompt),
                "timestamp": r.get("timestamp"),
                "input": prompt,
                "output": resp,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["model", "input", "output"])
    print("\nFamily counts before balancing:\n", df.family.value_counts().to_string())

    keep = df.family.value_counts()
    keep = keep[keep >= args.min_family].index
    df = df[df.family.isin(keep)]
    n = min(args.per_family, df.family.value_counts().min())
    df = df.groupby("family", group_keys=False).sample(n=n, random_state=args.seed).reset_index(drop=True)

    print(f"\nBalanced to {n} per family, {len(df)} rows total")
    print("\nFamily x domain:\n", pd.crosstab(df.family, df.domain).to_string())
    print("\nModels per family:\n", df.groupby("family").model.nunique().to_string())
    print("\nReasoning-variant share per family:\n", df.groupby("family").is_reasoning.mean().round(2).to_string())
    leak = df.output.str.contains(r"<think>|</think>|^\s*(Thinking|Reasoning)\b", case=False, regex=True)
    print("\nOutputs containing visible thinking traces per family:\n", df.assign(leak=leak).groupby("family").leak.sum().to_string())
    print("\nPrompts answered by 2 kept rows (matched pairs):", (df.prompt_id.value_counts() == 2).sum())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False, compression="zstd")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
