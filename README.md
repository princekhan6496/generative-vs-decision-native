# Generative LLM vs Decision-Native Model for AI Automation

Educational PyTorch research project comparing a decoder-only Mini-LLM with a decision-native Mini-Jev architecture inspired only by publicly documented TypeSafe/System One concepts. It is **not a reproduction of TypeSafe Jev**.

## What is implemented

- Mini-LLM: tokenizer, token embeddings, RoPE, causal multi-head attention, RMSNorm, SwiGLU, residual blocks, LM head, cross-entropy, AdamW, clipping, checkpoints, autoregressive generation, temperature, top-k and top-p.
- Mini-Jev: shared state encoder, typed question encoder, Choice/Score/Noul heads, simultaneous multi-question inference, supervised mixed-task training, confidence, and calibration utilities.
- Evaluation: accuracy, macro F1, precision, recall, Brier, ECE, NLL, reliability data, confidence histograms, latency, throughput, parameter counts and structured-output failure rate.
- Experimental controls: train-only tokenizer fitting, deterministic train/validation/test split, validation-only temperature scaling, repeated seeds, parameter matching, compute matching, and a lexical distribution-shift experiment.
- Separate TinyStories language-generation experiment.
- Optional `mini_jev/rlcd_experiment.py`: an explicitly labeled RLCD-inspired educational objective. It is **not TypeSafe RLCD**.

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Verification

```bash
pytest -q
python scripts/smoke_test.py
python scripts/tiny_train.py
```

The repository contains tests and smoke-training checks. Benchmark results are only written after an actual benchmark run; there are no fabricated final results.

## Data

The project supports BANKING77 and CLINC150. Download them with:

```bash
python scripts/download_data.py
```

Downloaded datasets are not bundled. Check the individual dataset licenses and terms before redistribution.

## Decision benchmark

BANKING77:

```bash
python experiments/benchmark.py \
  --dataset banking77 \
  --banking-train data/banking77_train.csv \
  --banking-test data/banking77_test.csv \
  --epochs 3 \
  --seeds 42,43,44 \
  --out results/banking77.json
```

Parameter-matched:

```bash
python experiments/benchmark.py --dataset banking77 --banking-train data/banking77_train.csv --banking-test data/banking77_test.csv --protocol parameter-matched --seeds 42,43,44 --out results/banking77_parameter_matched.json
```

Compute-matched:

```bash
python experiments/benchmark.py --dataset banking77 --banking-train data/banking77_train.csv --banking-test data/banking77_test.csv --protocol compute-matched --seeds 42,43,44 --out results/banking77_compute_matched.json
```

CLINC150:

```bash
python experiments/benchmark.py --dataset clinc150 --clinc-json data/clinc150.json --epochs 3 --seeds 42,43,44 --out results/clinc150.json
```

The benchmark fits the tokenizer on training data only, creates validation data from the training distribution, never tunes on the test set, and uses the same underlying examples and labels for both architectures. The LLM probability benchmark uses conditional option likelihoods; generated structured output is evaluated separately for malformed-output failures.

## Distribution shift

```bash
python experiments/ood.py --banking-train data/banking77_train.csv --banking-test data/banking77_test.csv --epochs 3 --out results/ood.json
```

The OOD test applies a fixed lexical transformation to the held-out test states. No tuning is performed on the shifted set.

## Language generation

Provide a local TinyStories text file, one story/example per line:

```bash
python experiments/tiny_stories.py --text data/tinystories.txt --epochs 3 --out results/tinystories.json
```

This experiment is separate from the decision benchmark. Do not combine its metrics with decision metrics.

## Interpretation

Do not declare a universal winner. The experiments answer narrower questions about which architecture is appropriate for fixed structured decisions, calibrated probabilities, parallel questions, and open-ended language generation.

## Public TypeSafe context

TypeSafe publicly describes System One models as decision-oriented models with typed probabilistic outputs, parallel sampling and RLCD. Its public material does not disclose the proprietary internal architecture in sufficient detail to reproduce it faithfully. Our Transformer encoder, RoPE, RMSNorm, SwiGLU, question encoder, supervised objective and temperature scaling are therefore **implementation choices**, not claims about Jev's internals.

References:
- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://typesafe.ai/
- https://evals.typesafe.ai/
- https://huggingface.co/datasets/PolyAI/banking77
- https://github.com/clinc/oos-eval
- https://huggingface.co/datasets/roneneldan/TinyStories
