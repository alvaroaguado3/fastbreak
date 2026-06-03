# Headline generation training

Solves the cold-start problem (no labeled headlines on day one) in three stages.

## 1. Bootstrap a synthetic dataset
The template engine encodes the feature→phrasing mapping. We sample plausible
game situations, render template headlines, and store `(prompt, completion)` pairs.

```bash
python training/generate_synthetic.py --n 4000 --multi-frac 0.3
# -> data/headlines_synth.jsonl
```

## 2. Fine-tune a LoRA adapter on a small base model
```bash
pip install -e ".[ml]"
python -m training.train_lora --data data/headlines_synth.jsonl \
    --base Qwen/Qwen2.5-0.5B --out checkpoints/headline-lora --epochs 3
```
LoRA (Hu et al., 2021) trains <1% of parameters, so this runs on one consumer
GPU in minutes and keeps inference latency low.

## 3. Close the loop with real feedback
Once live, `FeedbackStore` records which headlines humans actually reused.
Re-export those as high-weight training pairs (reused = positive) and re-run
step 2 nightly. The `SurpriseSelector.update_from_feedback` hook simultaneously
recalibrates the publish threshold online.

## Use the trained model
```bash
export FASTBREAK_GENERATOR=neural
export FASTBREAK_NEURAL_MODEL_PATH=checkpoints/headline-lora
```
`NeuralGenerator.available()` falls back to templates if the adapter or the
`ml` extras are missing, so the live system never hard-fails.

## Evaluation
- **Faithfulness:** does the headline's number/subject match the input flags?
  (regex check against the prompt — a hallucinated stat is an automatic fail.)
- **Reuse rate:** the production north-star — fraction of published headlines a
  human reshared. Tracked from `FeedbackStore.iter_labeled()`.
