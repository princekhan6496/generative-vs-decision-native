# Architecture

This document describes the architectures implemented in this repository.

> **Important:** The Mini-Jev architecture below is an original educational implementation inspired by publicly documented TypeSafe/System One behavior. It is **not** TypeSafe Jev's disclosed internal architecture.

## 1. Mini-LLM

High-level flow:

```text
tokens
  ↓
token embeddings
  ↓
RoPE
  ↓
N decoder blocks
  ↓
final RMSNorm
  ↓
vocabulary logits
```

The Mini-LLM is a decoder-only Transformer.

### Decoder block

Each decoder block follows:

```text
RMSNorm
  ↓
causal multi-head self-attention
  ↓
residual connection
  ↓
RMSNorm
  ↓
SwiGLU feed-forward network
  ↓
residual connection
```

### Causal attention

For a sequence of hidden states, learned projections produce:

```text
Q = XW_Q
K = XW_K
V = XW_V
```

The scaled attention operation is:

```text
Attention(Q, K, V)
  = softmax(QKᵀ / √d_head + causal_mask)V
```

The causal mask prevents a position from attending to future tokens.

### RoPE

Rotary positional embeddings rotate pairs of query/key coordinates by a position-dependent angle.

For an adjacent coordinate pair:

```text
(x_2i, x_2i+1)
```

the pair is rotated using sine/cosine values determined by position and frequency.

RoPE is an implementation choice of this Mini-LLM. It is not evidence that TypeSafe Jev uses RoPE.

### SwiGLU

The feed-forward block uses a SwiGLU-style gated transformation.

The exact dimensions and configuration are repository implementation parameters.

### Language-model objective

For a target token sequence, the model predicts the next token at each position.

The training objective is categorical cross-entropy over vocabulary tokens.

Optimization uses AdamW with gradient clipping.

### Generation

The Mini-LLM supports autoregressive generation.

At each step:

```text
previous tokens
      ↓
Transformer
      ↓
next-token logits
      ↓
temperature / top-k / top-p sampling
      ↓
sampled token
      ↓
append token and repeat
```

Because generation is autoregressive, later tokens depend on earlier generated tokens.

## 2. Mini-Jev

High-level flow:

```text
state
  ↓
token embeddings + positional representation
  ↓
Transformer encoder
  ↓
pooled shared state representation
  ↓
shared representation reused for each question
```

For each question:

```text
question instruction + learned type representation
                    ↓
              question encoder
                    ↓
        state representation + question representation
                    ↓
                 typed head
```

The three implemented typed heads are:

```text
Choice
Score
Noul
```

### Choice

Choice represents a selection from a predefined set of options.

The implementation computes a score for each candidate option and applies softmax:

```text
p_i = exp(z_i) / Σ_j exp(z_j)
```

The resulting vector is a probability distribution over the allowed options.

The application, rather than the model, owns any downstream thresholding or action logic.

### Score

Score represents an ordered rating over predefined levels.

The implementation produces a probability distribution over the ordered levels and computes an expected score:

```text
E[s] = Σ_i p_i s_i
```

The level set and numerical mapping are benchmark/model design choices in this repository.

They must not be presented as TypeSafe's internal scoring implementation.

### Noul

Noul represents a binary yes/no decision.

The implementation produces a binary logit and converts it to a probability with a sigmoid:

```text
p = sigmoid(z)
```

### Multi-question inference

A central feature of the Mini-Jev design is shared-state computation.

Instead of independently encoding the same state for every question, the model computes the shared state representation once:

```text
                         ┌─ question 1 → Choice
                         │
state → shared encoder ──┼─ question 2 → Noul
                         │
                         ├─ question 3 → Score
                         │
                         └─ question 4 → Choice
```

This is the implementation's basis for testing multi-question inference.

The measured latency benefit is empirical. It is not guaranteed and depends on implementation, hardware and workload.

## 3. Calibration

For probability calibration, the benchmark uses validation-only temperature scaling:

```text
p = softmax(z / T)
```

where `T` is fitted using validation outputs.

The fitted `T` is then frozen before test evaluation.

This is a standard calibration baseline in this repository.

It is **not** TypeSafe's RLCD method.

## 4. Mini-LLM decision evaluation

The Mini-LLM is fundamentally a text-generating model.

To compare its probabilities with the Mini-Jev decision heads, the benchmark:

1. Builds a structured decision prompt.
2. Lists the allowed candidate options.
3. Computes conditional log-likelihoods for candidate answers.
4. Normalizes the candidate scores into a probability distribution.

This produces a probability vector suitable for metrics such as Brier score and NLL.

It does not turn the Mini-LLM into a native Choice/Score/Noul model.

Generated structured output is evaluated separately for parse/mapping failures.

## 5. Shared equations

### Softmax

```text
p_i = exp(z_i) / Σ_j exp(z_j)
```

### Binary probability

```text
p = sigmoid(z)
```

### Score expectation

```text
E[s] = Σ_i p_i s_i
```

### Temperature scaling

```text
p = softmax(z / T)
```

### Cross-entropy

For a target class `y`:

```text
L = -log p_y
```

Choice/Score heads use categorical cross-entropy.

Noul uses binary cross-entropy with logits.

The Mini-LLM uses next-token cross-entropy.

## 6. Architecture boundary: public TypeSafe information vs this repository

### Publicly documented TypeSafe concepts

TypeSafe publicly describes System One models as:

- decision-oriented models for software/automation;
- models that return typed decisions;
- models whose decisions include probabilities/confidence;
- systems supporting Noul, Choice and Score decision primitives;
- systems designed to evaluate multiple questions against shared state;
- a model family using a new architecture, parallel sampler and RLCD training method.

### Repository implementation choices

This repository chooses:

- a decoder-only Transformer for Mini-LLM;
- a Transformer encoder for Mini-Jev;
- RoPE in Mini-LLM;
- RMSNorm;
- SwiGLU;
- a separate question representation;
- shared state encoding;
- Choice/Score/Noul heads;
- supervised mixed-task training;
- temperature scaling;
- BANKING77/CLINC150 experiments;
- a lexical OOD transformation;
- parameter and workload matching protocols.

The second list must never be presented as a description of Jev's hidden architecture.

## 7. Architectural limitation

Mini-Jev is intentionally small and educational.

It should therefore be described as:

> **“Mini-Jev, an educational decision-native architecture inspired by publicly documented Jev/System One concepts.”**

It should not be described as:

> “TypeSafe Jev reimplemented from scratch.”

The repository does not have access to TypeSafe's proprietary weights, internal architecture, training data, sampler implementation or RLCD implementation.
