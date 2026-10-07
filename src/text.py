"""Text utilities: tokenization, vocabularies, encoding, ablations, stylometric features."""
import re
from collections import Counter

import numpy as np
import pandas as pd

PAD, UNK, SEP = 0, 1, 2
SPECIALS = ["<pad>", "<unk>", "<sep>"]

# ---------------------------------------------------------------- tokenization
# Word tokens keep punctuation and newlines as tokens: formatting is part of the signal.
WORD_RE = re.compile(r"\n|\w+|[^\w\s]")


def word_tokens(text: str):
    return ["<nl>" if t == "\n" else t.lower() for t in WORD_RE.findall(text)]


def char_tokens(text: str):
    return list(text)


TOKENIZERS = {"word": word_tokens, "char": char_tokens}


class Vocab:
    def __init__(self, texts, level: str, max_size: int, min_freq: int):
        self.level = level
        self.tok = TOKENIZERS[level]
        counts = Counter(t for x in texts for t in self.tok(x))
        items = [t for t, c in counts.most_common(max_size) if c >= min_freq]
        self.itos = SPECIALS + items
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def from_itos(cls, itos, level):
        v = cls.__new__(cls)
        v.level, v.tok, v.itos = level, TOKENIZERS[level], list(itos)
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str, max_len: int):
        return [self.stoi.get(t, UNK) for t in self.tok(text)[:max_len]]


def encode_fields(vocab: Vocab, inputs, outputs, mode: str, max_len: int, input_share: float = 0.25):
    """mode: 'output' | 'input' | 'both'. For 'both', the input gets at most
    input_share of the budget so long prompts cannot crowd out the response."""
    seqs = []
    for x, y in zip(inputs, outputs):
        if mode == "output":
            s = vocab.encode(y, max_len)
        elif mode == "input":
            s = vocab.encode(x, max_len)
        else:
            xi = vocab.encode(x, int(max_len * input_share))
            s = xi + [SEP] + vocab.encode(y, max_len - len(xi) - 1)
        seqs.append(s if s else [UNK])
    return seqs


def field_text(df: pd.DataFrame, mode: str):
    """Plain-text view of the requested fields (for TF-IDF / feature models)."""
    if mode == "output":
        return df.output.tolist()
    if mode == "input":
        return df.input.tolist()
    return (df.input + " <sep> " + df.output).tolist()


# ---------------------------------------------------------------- ablations (RQ4)
MD_HEADER = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
MD_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M)
MD_EMPH = re.compile(r"(\*\*|__|\*|_|~~|`)")
MD_FENCE = re.compile(r"^```.*$", re.M)
MD_HR = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.M)
MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}.*$", re.M)
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")


def strip_markdown(t):
    for rx in (MD_FENCE, MD_HR, MD_TABLE_SEP):
        t = rx.sub("", t)
    t = MD_HEADER.sub("", t)
    t = MD_BULLET.sub("", t)
    t = MD_EMPH.sub("", t)
    return t.replace("|", " ")


def normalize_punct(t):
    t = t.replace("—", "-").replace("–", "-").replace("…", "...")
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return EMOJI.sub("", t)


def flatten_ws(t):
    return re.sub(r"\s+", " ", t).strip()


def fixed_length(t, n=600):
    return t[:n]


ABLATIONS = {
    "none": lambda t: t,
    "strip_markdown": strip_markdown,
    "normalize_punct": normalize_punct,
    "flatten_whitespace": flatten_ws,
    "fixed_length_600": fixed_length,
    "lowercase": str.lower,
    "all_structure_removed": lambda t: fixed_length(flatten_ws(normalize_punct(strip_markdown(t)))).lower(),
}


# ---------------------------------------------------------------- stylometric features (RQ4)
OPENERS = re.compile(r"^\s*(excellent|great|certainly|of course|sure|absolutely|okay|ok|alright|hello|hi|what a)\b", re.I)
FOLLOWUP = re.compile(r"(let me know|would you like|do you want|want me to|feel free|hope this helps|happy to help)[^\n]*$", re.I)


def stylometric(text: str) -> dict:
    words = re.findall(r"\w+", text)
    nw = max(len(words), 1)
    nc = max(len(text), 1)
    sents = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
    lines = text.split("\n")
    w200 = [w.lower() for w in words[:200]]
    return {
        "log_chars": np.log1p(len(text)),
        "log_words": np.log1p(len(words)),
        "avg_word_len": sum(map(len, words)) / nw,
        "avg_sent_words": nw / max(len(sents), 1),
        "ttr_first200": len(set(w200)) / max(len(w200), 1),
        "newlines_per_100w": 100 * text.count("\n") / nw,
        "blank_lines_per_100w": 100 * sum(1 for l in lines if not l.strip()) / nw,
        "headers_per_100w": 100 * len(MD_HEADER.findall(text)) / nw,
        "bullets_per_100w": 100 * len(MD_BULLET.findall(text)) / nw,
        "bold_per_100w": 100 * text.count("**") / 2 / nw,
        "code_fences": text.count("```") / 2,
        "table_rows_per_100w": 100 * sum(1 for l in lines if l.strip().startswith("|")) / nw,
        "hr_count": len(MD_HR.findall(text)),
        "latex": int(bool(re.search(r"\$[^$]+\$|\\\(|\\\[", text))),
        "emoji_per_1kc": 1000 * len(EMOJI.findall(text)) / nc,
        "emdash_per_1kc": 1000 * text.count("—") / nc,
        "endash_per_1kc": 1000 * text.count("–") / nc,
        "colon_per_1kc": 1000 * text.count(":") / nc,
        "semicolon_per_1kc": 1000 * text.count(";") / nc,
        "exclam_per_1kc": 1000 * text.count("!") / nc,
        "question_per_1kc": 1000 * text.count("?") / nc,
        "paren_per_1kc": 1000 * text.count("(") / nc,
        "curly_quote_per_1kc": 1000 * sum(text.count(c) for c in "“”’") / nc,
        "opener_affirmation": int(bool(OPENERS.search(text))),
        "ends_with_offer": int(bool(FOLLOWUP.search(text.strip()[-300:]))),
        "first_person_per_100w": 100 * sum(w.lower() in ("i", "i'm", "me", "my") for w in words) / nw,
        "you_per_100w": 100 * sum(w.lower() in ("you", "your") for w in words) / nw,
    }


def stylometric_frame(texts) -> pd.DataFrame:
    return pd.DataFrame([stylometric(t) for t in texts])
