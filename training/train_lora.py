"""LoRA fine-tuning harness for the headline generator.

Why LoRA + a small base model: the spec calls for low latency. A 0.5B-1.1B base
model with a small LoRA adapter (rank 8-16) trains on a single consumer GPU in
minutes and generates a 24-token headline in tens of milliseconds. LoRA
(Hu et al., 2021, arXiv:2106.09685) updates <1% of params, so iteration is cheap
and you can retrain nightly as feedback accumulates.

Usage:
  pip install -e ".[ml]"
  python training/generate_synthetic.py --n 4000
  python -m training.train_lora --data data/headlines_synth.jsonl \
      --base Qwen/Qwen2.5-0.5B --out checkpoints/headline-lora --epochs 3

Then set FASTBREAK_GENERATOR=neural and FASTBREAK_NEURAL_MODEL_PATH=checkpoints/headline-lora.
"""
from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/headlines_synth.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="checkpoints/headline-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args(argv)

    import torch
    from datasets import Dataset  # noqa: F401  (ensures extra installed)
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    from .dataset import build_hf_dataset

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = build_hf_dataset(args.data, tok, max_len=args.max_len)
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, learning_rate=args.lr,
        logging_steps=25, save_strategy="epoch", warmup_ratio=0.03,
        fp16=torch.cuda.is_available(), report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
