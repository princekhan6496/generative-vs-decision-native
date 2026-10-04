# Generative LLM vs Decision-Native Model for AI Automation

Educational PyTorch research project comparing a decoder-only **Mini-LLM** with a **Mini-Jev**, an educational decision-native architecture inspired only by publicly documented TypeSafe/System One concepts.

> **Important:** This project is **not a reproduction, reimplementation, or reverse engineering of TypeSafe Jev**. TypeSafe does not publicly disclose enough internal architectural or training details to support such a claim.

## What is implemented

### Mini-LLM

A small decoder-only Transformer implemented in PyTorch, including:

- train-only tokenizer fitting
- token embeddings
- RoPE positional encoding
- causal multi-head self-attention
- RMSNorm
- SwiGLU feed-forward layers
- residual connections
- vocabulary / language-model head
- next-token cross-entropy training
- AdamW optimization
- gradient clipping
- checkpoints
- autoregressive generation
- temperature, top-k and top-p sampling

For decision evaluation, the Mini-LLM also estimates option probabilities from normalized conditional log-likelihoods. This is an evaluation method, not a claim that the LLM has a native categorical decision head.

### Mini-Jev

An educational decision-native architecture with:

- shared state encoder
- typed question representation
- Choice, Score and Noul heads
- supervised mixed-task training
- multi-question inference from a shared state representation
- probability and confidence utilities
- validation-only temperature scaling for calibration

The architecture is an original implementation choice based on the public behavioral description of System One models. It does not claim to match Jev internally.

### Evaluation

The benchmark records:

- accuracy
- macro F1
- precision
- recall
- Brier score
- expected calibration error (ECE)
- negative log-likelihood (NLL)
- reliability-bin data
- confidence histograms
- latency
- throughput
- parameter counts
- structured-output failure rate

The decision benchmark also compares parallel multi-question inference with a serial per-question baseline.

### Experimental controls

The repository includes:

- train-only tokenizer fitting
- deterministic train/validation/test construction
- repeated random seeds
- parameter-matched experiments
- a workload-matched `compute-matched` protocol
- validation-only temperature scaling
- a fixed lexical distribution-shift experiment
- separate TinyStories language-generation evaluation

The `compute-matched` protocol matches the number of training examples, batch size and optimizer steps. It does **not** perform exact FLOP matching. Actual tokens processed and measured runtime are reported so the limitation is visible.

An optional `mini_jev/rlcd_experiment.py` provides an explicitly labeled **RLCD-inspired educational objective**. It is not TypeSafe's RLCD algorithm.

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Verification

Run the repository checks before running the larger experiments:

```bash
pytest -q
python scripts/smoke_test.py
python scripts/tiny_train.py
```

These checks verify the implementation and basic training path. Benchmark result files are produced by benchmark execution; they are not hard-coded as evidence.

## Data

The project supports:

- BANKING77
- CLINC150

Download the supported datasets with:

```bash
python scripts/download_data.py
```

Downloaded datasets are not bundled with the repository. Review the individual dataset licenses and terms before redistribution.

## Decision benchmark

### BANKING77

Standard repeated-seed benchmark:

```bash
python experiments/benchmark.py --dataset banking77 --banking-train data/banking77_train.csv --banking-test data/banking77_test.csv --epochs 3 --seeds 42,43,44 --out results/banking77.json
```

### Parameter-matched

The Mini-LLM configuration is kept fixed. A separate Mini-Jev configuration is searched so that its parameter count is as close as practical to the Mini-LLM configuration.

```bash
python experiments/benchmark.py --dataset banking77 --banking-train data/banking77_train.csv --banking-test data/banking77_test.csv --protocol parameter-matched --seeds 42,43,44 --out results/banking77_parameter_matched.json
```

Parameter matching controls model size approximately; it does not make the architectures computationally identical.

### `compute-matched`

The repository's `compute-matched` protocol uses the same batch size, optimizer-step count and number of training examples for the two models.

```bash
python experiments/benchmark.py --dataset banking77 --banking-train data/banking77_train.csv --banking-test data/banking77_test.csv --protocol compute-matched --seeds 42,43,44 --out results/banking77_compute_matched.json
```

This is a **controlled training-workload match**, not an exact FLOP match. The benchmark records tokens processed and measured training time so that differences in actual computational work remain visible.

### CLINC150

```bash
python experiments/benchmark.py --dataset clinc150 --clinc-json data/clinc150.json --epochs 3 --seeds 42,43,44 --out results/clinc150.json
```

### Benchmark protocol

The benchmark:

1. Builds the underlying records once.
2. Derives the same state/question/ground-truth examples for both architectures.
3. Creates training and validation data from the training distribution and uses the public test split as held-out test data.
4. Fits the tokenizer on training information only.
5. Uses the same requested random seeds for both architectures.
6. Does not tune on the test set.
7. Evaluates Mini-LLM option probabilities using conditional option likelihoods.
8. Evaluates generated structured output separately for malformed-output failures.

For the common decision benchmark, derived routing and risk tasks are deterministic benchmark constructions; they are **not additional human-annotated BANKING77 labels**.

## Distribution shift

Run the lexical distribution-shift experiment with:

```bash
python experiments/ood.py --banking-train data/banking77_train.csv --banking-test data/banking77_test.csv --epochs 3 --out results/ood.json
```

The experiment applies a fixed lexical transformation to held-out test states and evaluates without adapting or tuning the models on the transformed set.

The comparison records performance and calibration degradation under this controlled shift.

## Language generation

Provide a local TinyStories text file, with one story/example per line:

```bash
python experiments/tiny_stories.py --text data/tinystories.txt --epochs 3 --out results/tinystories.json
```

TinyStories is a separate next-token language-modeling experiment. Its metrics must not be combined with the decision-task metrics.

## Interpreting results

The project is designed to answer narrower research questions, not to establish a universal winner.

Examples of questions the experiments can address:

- How does a generative decoder perform when structured decisions are represented as text?
- How does a decision-native architecture perform on fixed Choice/Score/Noul tasks?
- How do their probability estimates compare?
- How stable are their results across random seeds?
- What is the measured cost of training and inference in these implementations?
- Does shared-state multi-question inference provide a measurable latency benefit?
- How do the architectures behave under the selected lexical distribution shift?
- What changes when the task is open-ended language generation rather than fixed decisions?

A result from this repository should be described as a result of **these educational implementations and protocols**, not as a measurement of TypeSafe's proprietary internal architecture.

## Public TypeSafe context

TypeSafe publicly describes System One models as a class of models intended for decisions inside software. Its public material describes Jev as producing typed decisions with probabilities and confidence, supporting question types including Noul, Choice and Score, and evaluating multiple questions against shared state. TypeSafe also describes a new architecture, a parallel sampler, and a training method called Reinforcement Learning for Calibrated Decisions (RLCD).

The public material does not disclose sufficient internal implementation detail to reproduce Jev faithfully. Therefore, the Transformer encoder, RoPE, RMSNorm, SwiGLU, question encoder, supervised objectives, temperature scaling and benchmark design in this repository are **our implementation choices**.

See `VERIFIED_VS_ASSUMED.md` for the exact boundary between public facts and project assumptions.

## References

- TypeSafe: https://typesafe.ai/
- TypeSafe announcement: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe workflow evaluations: https://evals.typesafe.ai/
- BANKING77: https://huggingface.co/datasets/PolyAI/banking77
- CLINC OOS evaluation repository: https://github.com/clinc/oos-eval
- TinyStories: https://huggingface.co/datasets/roneneldan/TinyStories

## Research integrity

This repository should only report benchmark numbers that were produced by running the corresponding experiment. Missing or unrun experiments are **unmeasured**, not assumed, estimated or filled with expected values.

The project intentionally distinguishes:

- public facts about TypeSafe/ System One / Jev
- architecture and training choices made in this repository
- measured experimental results
- limitations of the experimental controls
