"""Neural headline generator (LoRA-adapted small causal LM).

Design for low latency:
  - Small base model (e.g. Qwen2.5-0.5B / TinyLlama-1.1B) + LoRA adapter, so
    the merged model is tiny and can run on CPU or a modest GPU.
  - Greedy / low-temperature short generations (<= ~24 new tokens).
  - A structured prompt built from the SAME features the templates use, so the
    model learns "features -> headline" rather than free-form hallucination.

This module imports torch/transformers lazily and falls back gracefully: if the
extras aren't installed or the adapter is missing, `available()` returns False
and the pipeline keeps using templates. That keeps the core dependency-free.

Train an adapter with: python -m training.train_lora  (see training/README.md)
"""
from __future__ import annotations

import time

from ..models import HeadlineCandidate, StatFlag


def build_prompt(flags: list[StatFlag]) -> str:
    """Serialize flags into the structured prompt the model is trained on."""
    lines = ["### Game situation:"]
    ev = flags[0].event
    ctx = ev.context
    lines.append(
        f"{ctx.away_team} @ {ctx.home_team} | {ctx.game_type} | "
        f"Q{ctx.quarter} {ctx.clock_seconds // 60}:{ctx.clock_seconds % 60:02d} | {ctx.venue}"
    )
    lines.append("### Notable stats:")
    for f in flags:
        lines.append(
            f"- {f.event.subject_name}: {int(f.event.value)} {f.stat} "
            f"(p{int(f.percentile * 100)}, {f.surprisal_bits:.1f} bits, ctx={f.matched_context})"
        )
    lines.append("### Headline:")
    return "\n".join(lines)


class NeuralGenerator:
    source = "neural"

    def __init__(self, model_path: str, base_model: str | None = None,
                 max_new_tokens: int = 24, device: str | None = None):
        self.model_path = model_path
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._model = None
        self._tok = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        import os
        return os.path.exists(self.model_path)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg_base = self.base_model
        if cfg_base is None:
            # adapter_config.json records its base model
            import json, os
            ac = os.path.join(self.model_path, "adapter_config.json")
            cfg_base = json.load(open(ac))["base_model_name_or_path"]

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(cfg_base)
        base = AutoModelForCausalLM.from_pretrained(
            cfg_base, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self._model = PeftModel.from_pretrained(base, self.model_path).to(self.device).eval()

    def generate(self, flags: list[StatFlag], n: int = 4) -> list[HeadlineCandidate]:
        if not flags:
            return []
        self._load()
        import torch

        prompt = build_prompt(flags)
        inputs = self._tok(prompt, return_tensors="pt").to(self.device)
        out: list[HeadlineCandidate] = []
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                num_return_sequences=n,
                pad_token_id=self._tok.eos_token_id,
            )
        latency = (time.perf_counter() - t0) * 1000.0 / max(n, 1)
        for seq in gen:
            text = self._tok.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            text = text.strip().split("\n")[0].strip()
            if text:
                out.append(HeadlineCandidate(text=text, flags=flags,
                                             source=self.source, gen_latency_ms=latency))
        return out
