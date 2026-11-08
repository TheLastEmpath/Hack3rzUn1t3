"""Prompt sets: harmful (AdvBench) vs harmless (Alpaca), same style, disjoint splits.

The refusal direction is only a *refusal* direction if the two sets differ in
harm and little else, so the harmless set is filtered to single-instruction
prompts of similar length to AdvBench goals.
"""

import csv
import hashlib
import json
import random
import urllib.request
from pathlib import Path

# Pinned so a re-run (and the stage 1b exact-match check) sees the same text.
ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
ALPACA_URL = (
    "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/"
    "12567cabf869d7c92e573c7c783905fc160e9639/alpaca_data_cleaned.json"
)


def fetch(url: str, dest: Path) -> Path:
    """Download url to dest unless already there. Returns dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    return dest


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_advbench(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [row["goal"].strip() for row in csv.DictReader(f)]


def filter_alpaca(rows: list[dict], min_len: int = 30, max_len: int = 150) -> list[str]:
    """Keep plain instructions: no extra `input` field, AdvBench-like length, deduped."""
    seen, out = set(), []
    for row in rows:
        if row.get("input", "").strip():
            continue
        text = row["instruction"].strip()
        if not (min_len <= len(text) <= max_len) or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(text)
    return out


def load_alpaca(path: Path) -> list[str]:
    return filter_alpaca(json.loads(path.read_text(encoding="utf-8")))


def split(prompts: list[str], n_train: int, n_val: int, seed: int):
    """Seeded shuffle, then disjoint train / validation slices."""
    if len(prompts) < n_train + n_val:
        raise ValueError(f"need {n_train + n_val} prompts, have {len(prompts)}")
    shuffled = list(prompts)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val]


def chat_texts(tokenizer, prompts: list[str]) -> list[str]:
    """Apply the chat template with the assistant turn left open."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
        )
        for p in prompts
    ]
