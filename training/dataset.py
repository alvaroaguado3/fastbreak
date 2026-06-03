"""Dataset loading + tokenization for causal-LM fine-tuning.

Builds (prompt + completion) sequences with the loss masked on the prompt tokens
so the model only learns to produce the headline, not to echo the situation.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_pairs(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_hf_dataset(path, tokenizer, max_len: int = 256):
    """Return a tokenized `datasets.Dataset` with prompt-masked labels."""
    from datasets import Dataset

    rows = load_pairs(path)

    def tok(example):
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        comp_ids = tokenizer(example["completion"] + tokenizer.eos_token,
                             add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + comp_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + comp_ids)[:max_len]  # mask the prompt
        attn = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    ds = Dataset.from_list(rows)
    return ds.map(tok, remove_columns=ds.column_names)
