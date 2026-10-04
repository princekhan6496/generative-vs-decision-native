# Architecture

## 1. Architectural Research Motivation

The central architectural question is:

> **What changes when a decision is represented as generated text versus an explicit typed probability distribution?**

The project therefore implements two deliberately different architectures.

### Mini-LLM

```text
state + decision prompt
        ↓
decoder-only Transformer
        ↓
text / token probabilities
        ↓
decision
```

### Mini-Jev

```text
state + typed question
        ↓
shared state representation
        ↓
typed decision head
        ↓
probability distribution
```

The Mini-Jev design is an educational architecture inspired by publicly documented TypeSafe/System One behavior. It is not TypeSafe Jev's disclosed internal architecture.

---

# 2. Mini-LLM Architecture

## 2.1 High-level pipeline

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

The model is a decoder-only Transformer.

Its primary objective is next-token prediction.

---

# 3. Mini-LLM Decoder Block

Each decoder block follows:

```text
Input
  │
  ├── RMSNorm
  │
  ├── Causal Multi-Head Self-Attention
  │
  └── Residual Add
       │
       ├── RMSNorm
       │
       ├── SwiGLU
       │
       └── Residual Add
              ↓
            Output
```

## 3.1 Attention

For hidden states `X`:

```text
Q = XW_Q
K = XW_K
V = XW_V
```

Scaled causal attention is:

```text
Attention(Q,K,V)
= softmax(QKᵀ / √d_head + causal_mask)V
```

The causal mask prevents a token from using future tokens.

## 3.2 RoPE

RoPE rotates query/key coordinate pairs according to token position and frequency.

For an adjacent pair:

```text
(x_2i, x_2i+1)
```

the coordinates are rotated using position-dependent sine/cosine values.

RoPE is an implementation choice of Mini-LLM.

It is **not evidence that TypeSafe Jev uses RoPE**.

## 3.3 RMSNorm

RMSNorm normalizes hidden representations before the attention and feed-forward transformations according to the repository's implementation.

## 3.4 SwiGLU

The feed-forward component uses a gated SwiGLU-style transformation.

Its dimensions are implementation parameters.

---

# 4. Mini-LLM Training

The Mini-LLM predicts the next token at each position.

The loss is categorical cross-entropy:

```text
L = -log p(y)
```

Optimization uses:

- AdamW
- gradient clipping
- checkpoints

The model can generate autoregressively:

```text
context
  ↓
Transformer
  ↓
next-token logits
  ↓
temperature / top-k / top-p
  ↓
sample token
  ↓
append token
  ↓
repeat
```

This makes the architecture flexible for language generation, but structured decisions must be represented through text.

---

# 5. Mini-Jev Architecture

## 5.1 High-level pipeline

```text
state
  ↓
token embeddings + positional representation
  ↓
Transformer encoder
  ↓
pooled shared state representation
  ↓
shared representation reused across questions
```

For each typed question:

```text
question instruction
        +
question/type representation
        ↓
question representation
        ↓
shared state + question representation
        ↓
typed head
```

The three heads are:

```text
Choice
Score
Noul
```

---

# 6. Choice Head

Choice represents a decision among predefined options.

Conceptually:

```text
shared state
     +
question
     ↓
option scores
     ↓
softmax
     ↓
P(option 1), ..., P(option n)
```

The probability distribution is:

```text
p_i = exp(z_i) / Σ_j exp(z_j)
```

The output is directly tied to the allowed decision space.

The application owns any downstream thresholds and actions.

---

# 7. Score Head

Score represents an ordered decision.

The model produces probabilities over predefined levels:

```text
low
medium
high
```

The expected score is:

```text
E[s] = Σ_i p_i s_i
```

The levels and numerical mapping are benchmark/model choices in this repository.

They are not claims about TypeSafe's internal Score implementation.

---

# 8. Noul Head

Noul is implemented as a binary decision:

```text
yes / no
```

The model produces a binary logit and probability:

```text
p = sigmoid(z)
```

This gives the model a direct representation for binary automation decisions.

---

# 9. Shared-State Multi-Question Inference

The main architectural difference being tested is the reuse of a state representation.

Instead of processing the same state independently for every question:

```text
state → encoder → question 1
state → encoder → question 2
state → encoder → question 3
state → encoder → question 4
```

Mini-Jev computes:

```text
state
  ↓
shared encoder
  ↓
shared representation
  ├── question 1 → typed head
  ├── question 2 → typed head
  ├── question 3 → typed head
  └── question 4 → typed head
```

This is the basis of the multi-question latency experiment.

A speedup is not assumed.

It must be measured because actual performance depends on hardware, batch size, sequence length and implementation overhead.

---

# 10. Decision Probability Comparison

The architectures produce probabilities differently.

## Mini-Jev

Probability is native to the typed head:

```text
state + question
       ↓
typed head
       ↓
probability distribution
```

## Mini-LLM

Probability is estimated from candidate answer likelihoods:

```text
decision prompt
       ↓
candidate option A → log likelihood
candidate option B → log likelihood
...
       ↓
normalize candidate likelihoods
       ↓
probability distribution
```

This gives both models a comparable probability representation for Brier and NLL.

However:

> **The Mini-LLM does not become a native decision model through this evaluation procedure.**

---

# 11. Calibration

The benchmark uses validation-only temperature scaling:

```text
p = softmax(z / T)
```

`T` is fitted on validation outputs.

It is then frozen and applied to the test set.

This is a calibration baseline.

It is **not TypeSafe RLCD**.

---

# 12. Architectural Comparison

| Property | Mini-LLM | Mini-Jev |
|---|---|---|
| Primary output | Tokens/text | Typed decision |
| Core architecture | Decoder Transformer | Encoder + typed heads |
| Native categorical decision head | No | Yes |
| Open-ended generation | Yes | Not the goal |
| Choice output | Text / likelihood evaluation | Direct probability distribution |
| Score output | Text / likelihood evaluation | Direct probability distribution |
| Noul output | Text / likelihood evaluation | Direct binary probability |
| Shared state across questions | Not the primary design | Yes |
| Structured output | Generated | Native |
| Calibration baseline | Temperature scaling | Temperature scaling |

This table describes the repository implementations only.

---

# 13. Why Compare These Architectures?

The comparison is not simply:

> “Which model gets the higher accuracy?”

It investigates whether the output representation itself changes measurable properties.

The research dimensions are:

```text
decision quality
      +
probability quality
      +
seed stability
      +
structured-output reliability
      +
training cost
      +
multi-question inference
```

This makes the project an architecture study rather than only a classification exercise.

---

# 14. Architectural Interpretation of the Results

The completed parameter-matched experiments showed:

### Mini-LLM

- slightly higher mean intent accuracy;
- higher mean macro F1;
- slightly better mean Brier score;
- substantially lower measured training time;
- considerably larger seed-to-seed variation.

### Mini-Jev

- slightly lower mean intent accuracy;
- lower mean macro F1;
- better mean NLL;
- much smaller seed-to-seed variation;
- substantially higher measured training time in this implementation;
- explicit typed probability outputs.

Therefore, the architecture comparison suggests:

> **Generative and decision-native representations produce different trade-offs even when parameter counts are nearly identical.**

---

# 15. Public TypeSafe Boundary

Publicly documented TypeSafe/System One concepts include:

- decision-oriented models;
- typed decisions;
- probability/confidence outputs;
- Noul;
- Choice;
- Score;
- multiple questions over shared state;
- a new architecture;
- a parallel sampler;
- RLCD.

The internal details of Jev are not sufficiently disclosed to establish that it uses:

- this Transformer encoder;
- this number of layers;
- this hidden size;
- this attention implementation;
- RoPE;
- RMSNorm;
- SwiGLU;
- this question encoder;
- these heads internally;
- this optimizer;
- this training objective;
- this sampler implementation.

Those are repository choices.

---

# 16. Correct Architecture Claim

Use:

> **“Mini-Jev, an educational decision-native architecture inspired by publicly documented Jev/System One concepts.”**

Do not use:

> “TypeSafe Jev implemented from scratch.”

or:

> “A reproduction of Jev.”

The repository has no access to TypeSafe's proprietary weights, internal architecture, training data, production inference stack or RLCD implementation.

---

# 17. Optional RLCD-Inspired Experiment

The repository contains an optional educational objective described as RLCD-inspired.

Its purpose is to explore the general research idea of optimizing calibrated decisions.

It must be described as:

> **RLCD-inspired educational objective**

and not:

> TypeSafe RLCD

because the actual TypeSafe RLCD algorithm is not publicly specified in sufficient detail for reproduction.
