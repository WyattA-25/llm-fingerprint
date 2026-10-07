"""
Step 2: clean the raw dataset and create grouped splits.

    python -m src.preprocess --in data/dataset.parquet --out data/processed.parquet

What it does
- Strips visible reasoning traces (<think>...</think>) so "has a thinking block"
  can't act as a label shortcut.
- Masks self-identification and vendor names (e.g. "I'm Claude, made by Anthropic")
  in BOTH input and output. Otherwise the classifier learns a name lookup, not a style.
  The unmasked text is kept in *_raw columns for the RQ4 "no masking" ablation.
- Adds analysis flags: is_reasoning, self_id (output named its own family), refusal.
- Splits 70/15/15 grouped by prompt_id, so the same prompt never appears in two splits.
"""
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

THINK = re.compile(r"<think>.*?</think>", re.S | re.I)

# Vendor/model names. "Google" and "Meta" are deliberately excluded: they are common
# content words (Google Sheets, meta-analysis) and masking them would destroy content.
NAME_PATTERNS = {
    "GPT": r"chat\s?gpt|gpt-?\d[\w.\-]*|\bgpt\b|open\s?ai|\bo[134](?:-mini)?\b",
    "Claude": r"\bclaude\b|anthropic",
    "Gemini": r"\bgemini\b|\bbard\b|deepmind",
    "Llama": r"\bllama[\w.\-]*|meta\s+ai",
    "Qwen": r"\bqwen[\w.\-]*|\bqwq\b|tongyi|alibaba\s+cloud",
    "DeepSeek": r"deep\s?seek",
    "Mistral": r"\bmistral\b|\bmagistral\b|le\s+chat",
    "Grok": r"\bgrok\b|\bxai\b|\bx\.ai\b",
}
ALL_NAMES = re.compile("|".join(f"(?:{p})" for p in NAME_PATTERNS.values()), re.I)
FAMILY_NAME = {k: re.compile(v, re.I) for k, v in NAME_PATTERNS.items()}

REASONING = re.compile(r"(?<!no-)thinking|deepseek-r1|qwq|magistral|^o\d|grok-3-mini|qwen3-(?:235b-a22b|30b-a3b)$", re.I)
REFUSAL = re.compile(r"^\s*(?:I'm sorry|I am sorry|Sorry,|I can't|I cannot|I can’t|I won't|I'm not able|I am not able|I'm unable)", re.I)

MASK = "[MODEL]"


def mask(text: str) -> str:
    return ALL_NAMES.sub(MASK, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/dataset.parquet")
    ap.add_argument("--out", default="data/processed.parquet")
    ap.add_argument("--seed", type=int, default=448)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    n0 = len(df)

    n_think = int(df.output.str.contains(THINK).sum())
    df["output_raw"] = df["output"].str.replace(THINK, "", regex=True).str.strip()
    df["input_raw"] = df["input"].str.strip()
    df = df[df.output_raw.str.len() >= 20].copy()

    df["is_reasoning"] = df.model.str.contains(REASONING)
    df["refusal"] = df.output_raw.str.contains(REFUSAL)
    df["self_id"] = [bool(FAMILY_NAME[f].search(o)) for f, o in zip(df.family, df.output_raw)]
    df["prompt_names_model"] = df.input_raw.str.contains(ALL_NAMES)

    df["input"] = df.input_raw.map(mask)
    df["output"] = df.output_raw.map(mask)

    # Grouped 70/15/15 split by prompt
    idx = np.arange(len(df))
    g = df.prompt_id.values
    tr, rest = next(GroupShuffleSplit(1, test_size=0.30, random_state=args.seed).split(idx, groups=g))
    va_rel, te_rel = next(GroupShuffleSplit(1, test_size=0.50, random_state=args.seed).split(rest, groups=g[rest]))
    split = np.empty(len(df), dtype=object)
    split[tr], split[rest[va_rel]], split[rest[te_rel]] = "train", "val", "test"
    df["split"] = split

    # Matched pair = a prompt answered by two different families, both kept.
    fam_per_prompt = df.groupby("prompt_id").family.nunique()
    df["matched_pair"] = df.prompt_id.map(fam_per_prompt).eq(2)

    df = df.reset_index(drop=True)
    df.to_parquet(args.out, index=False)

    print(f"Rows: {n0} -> {len(df)} after dropping near-empty outputs")
    print("\nSplit sizes:\n", df.split.value_counts().to_string())
    assert df.groupby("prompt_id").split.nunique().max() == 1, "prompt leaked across splits"
    print("\nPer-family flags (share):")
    print(df.groupby("family")[["is_reasoning", "refusal", "self_id", "prompt_names_model"]].mean().round(3).to_string())
    print("\nOutputs that contained <think> traces (stripped):", n_think)
    print("Matched-pair rows:", int(df.matched_pair.sum()))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
