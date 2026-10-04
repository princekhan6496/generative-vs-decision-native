# Verified vs Assumed

## Purpose

This document defines the boundary between:

1. what is publicly documented by TypeSafe;
2. what this repository independently implements;
3. what the experiments actually demonstrate.

This boundary is essential because the project is inspired by TypeSafe/System One concepts but does not reproduce TypeSafe Jev.

---

# 1. Publicly Documented TypeSafe Concepts

The following are presented as public TypeSafe information.

## System One

TypeSafe presents System One models as models designed around decisions inside software and automation.

## Jev

Jev is presented as TypeSafe's first public System One model.

## Typed decisions

TypeSafe describes Jev/System One around structured decision primitives rather than ordinary unrestricted text generation.

Publicly documented decision types include:

### Noul

A binary yes/no decision.

### Choice

A decision among predefined choices with probabilistic/confidence information.

### Score

A decision represented using a score/rating space with probabilistic/confidence information.

## Shared state and multiple questions

Public material describes evaluating multiple questions against the same state and emphasizes parallel evaluation/sampling.

## Architecture and training

TypeSafe states that Jev uses a new model architecture and a parallel sampler.

TypeSafe also names its training method:

> **Reinforcement Learning for Calibrated Decisions (RLCD)**

The public material does not disclose enough information to reconstruct the exact proprietary architecture or RLCD algorithm.

---

# 2. What Is Not Publicly Established

The project must not state that TypeSafe Jev internally uses any specific implementation detail unless a primary TypeSafe source explicitly establishes it.

The following are **not established by the public information used for this project**:

- exact number of layers;
- exact hidden dimension;
- exact attention design;
- exact positional encoding;
- exact tokenizer;
- exact parameter count;
- exact weights;
- exact optimizer;
- exact learning-rate schedule;
- exact training dataset;
- exact sampler implementation;
- exact RLCD objective;
- exact production inference architecture.

Therefore, this repository does not claim any of these properties for Jev.

---

# 3. Our Implementation

The following are independent choices made for Mini-Jev:

- Transformer encoder for shared state representation;
- token embeddings;
- positional representation;
- pooled state representation;
- separate question representation;
- learned question/type representation;
- Choice head;
- Score head;
- Noul head;
- supervised mixed-task training;
- cross-entropy / binary cross-entropy losses;
- AdamW;
- gradient clipping;
- temperature scaling;
- BANKING77;
- CLINC150;
- derived routing task;
- derived risk task;
- derived security Noul task;
- conditional option likelihoods for Mini-LLM probability evaluation;
- lexical distribution-shift experiment;
- parameter-matching search;
- workload-matching protocol;
- serial versus shared-state latency experiment;
- TinyStories language-generation experiment.

These choices define **Mini-Jev**, not TypeSafe Jev.

---

# 4. Research Claim

The legitimate research claim is:

> **This project studies an educational decision-native architecture inspired by publicly documented TypeSafe/System One concepts and compares it with a small decoder-only generative LLM under controlled experimental conditions.**

The project does not claim:

> “We reproduced Jev.”

---

# 5. What the Experiments Actually Establish

The completed parameter-matched BANKING77 runs establish observations about the two implementations in this repository.

They show:

- Mini-LLM mean intent accuracy: **18.67%**
- Mini-Jev mean intent accuracy: **17.09%**
- Mini-LLM intent accuracy sample SD: **5.05 percentage points**
- Mini-Jev intent accuracy sample SD: **0.70 percentage points**
- Mini-LLM mean macro F1: **14.88%**
- Mini-Jev mean macro F1: **12.58%**
- Mini-LLM mean Brier: **0.915**
- Mini-Jev mean Brier: **0.925**
- Mini-LLM mean NLL: **3.435**
- Mini-Jev mean NLL: **2.909**
- Mini-LLM mean training time: **124.83 seconds**
- Mini-Jev mean training time: **1667.12 seconds**

The appropriate conclusion is:

> **Mini-LLM achieved slightly higher average task performance, while Mini-Jev showed substantially greater seed stability and lower mean NLL. Mini-LLM also had substantially lower measured training time in this implementation.**

These are observations about this experiment.

They are not claims about TypeSafe's production Jev.

---

# 6. Low Accuracy Boundary

The measured BANKING77 intent accuracy is low for both educational models.

This must be acknowledged.

The project does not claim state-of-the-art BANKING77 classification.

The research contribution is the controlled comparison of:

- decision representation;
- probabilistic behavior;
- seed stability;
- structured output;
- multi-question inference;
- measured computational cost.

Therefore:

> **Low absolute accuracy is a limitation of the current educational implementations, not evidence that the models achieve production-level BANKING77 performance.**

Likewise, the results should not be inflated into claims of production readiness.

---

# 7. Stability Claim Boundary

The three seeds show:

```text
Mini-LLM: 23.73%, 13.64%, 18.64%
Mini-Jev: 17.86%, 16.92%, 16.49%
```

Therefore the correct statement is:

> **Mini-Jev showed greater seed stability in these three experiments.**

Do not write:

> “Decision-native models are always more stable.”

The experiment does not establish that universal claim.

---

# 8. Computational Claim Boundary

Parameter matching produced approximately:

```text
Mini-LLM: 403,104
Mini-Jev: 403,818
```

The difference is approximately 0.18%.

However:

> **Equal parameter count does not mean equal FLOPs, memory traffic, implementation overhead or runtime.**

The measured training-time difference is therefore a result of these implementations and their training procedures.

It must not be generalized to TypeSafe Jev.

The `compute-matched` protocol also does not perform exact FLOP matching.

It matches:

- batch size;
- optimizer steps;
- examples seen.

---

# 9. RLCD Boundary

TypeSafe publicly names:

> **Reinforcement Learning for Calibrated Decisions (RLCD)**

The repository's optional experiment is only:

> **an RLCD-inspired educational objective**

It is not:

- TypeSafe RLCD;
- an implementation of TypeSafe RLCD;
- a reproduction of Jev's training algorithm.

The main benchmark's temperature scaling is also not RLCD.

---

# 10. Correct Terminology

### Preferred

> **Mini-Jev, an educational decision-native architecture inspired by publicly documented Jev/System One concepts.**

### Acceptable

> **A decision-native educational model inspired by the public behavioral description of TypeSafe's Jev.**

### Avoid

> reproduction of Jev

> reimplementation of Jev

> reverse-engineered Jev

> TypeSafe architecture recreated

> our version of Jev

> Jev uses Transformer X

unless directly supported by a primary TypeSafe source.

---

# 11. Public Behavior vs Repository Implementation

A useful separation is:

### Publicly documented concept

```text
state + typed questions
          ↓
typed probabilistic decisions
```

### Repository implementation

```text
state
 ↓
educational Transformer encoder
 ↓
shared representation
 +
question representation
 ↓
Choice / Score / Noul heads
 ↓
probabilities
```

The second diagram is **our design**.

It is not a hidden architecture diagram of Jev.

---

# 12. Source Boundary

Primary sources used for the TypeSafe context:

- https://typesafe.ai/
- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://api.typesafe.ai/docs
- https://api.typesafe.ai/redoc
- https://evals.typesafe.ai/

Dataset sources are separate:

- BANKING77
- CLINC OOS
- TinyStories

Dataset licenses and terms must be followed independently.

---

# 13. Final Research Position

The strongest defensible position of this project is:

> **We implemented a small generative LLM and an independently designed decision-native model inspired by public System One concepts, evaluated them on the same structured decision problems, and found different trade-offs in mean accuracy, probabilistic quality, seed stability and measured training cost.**

That is the research claim.

It is intentionally narrower than claiming to have reproduced or benchmarked TypeSafe Jev itself.
