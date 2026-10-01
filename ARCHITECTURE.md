# Architecture

## Mini-LLM

`tokens -> token embeddings -> RoPE -> N decoder blocks -> final RMSNorm -> vocabulary logits`

Each decoder block is:

`RMSNorm -> causal multi-head self-attention -> residual -> RMSNorm -> SwiGLU -> residual`

For attention, each head uses learned Q/K/V projections. RoPE rotates adjacent Q/K pairs using position-dependent sine/cosine values. The upper triangle of the attention score matrix is masked with negative infinity before softmax.

## Mini-Jev

`state -> token embeddings + positional embeddings -> Transformer encoder -> pooled shared state representation`

Each typed question has its own representation from instruction tokens plus a learned type embedding.

The state representation is reused for every question in the same call:

`shared state + question representation -> typed head`

- Choice: scores each predefined option and applies softmax.
- Score: scores learned ordered levels, applies softmax, then computes the probability-weighted expected score.
- Noul: produces one logit and applies sigmoid.

The application owns thresholds and external actions. The model returns probabilities/decisions and does not execute tools or side effects.

## Equations

Scaled attention:

`Attention(Q,K,V) = softmax((QK^T / sqrt(d_head)) + causal_mask)V`

RoPE for each adjacent pair `(x_2i, x_2i+1)` rotates the pair by the position/frequency angle.

Softmax:

`p_i = exp(z_i) / sum_j exp(z_j)`

Score expectation:

`E[s] = sum_i p_i * s_i`

Choice/Score losses use categorical cross entropy. Noul uses binary cross entropy with logits.

Temperature scaling:

`p = softmax(z / T)`

`T` is fitted on validation data only in the benchmark.

## Verified versus implementation choice

Verified from TypeSafe's public descriptions: System One is a decision-oriented model class; Jev exposes typed decision primitives and probabilistic/confidence outputs; multiple questions can be evaluated in parallel; TypeSafe describes a new architecture, parallel sampler and RLCD training method.

Implementation choices in this repository: Transformer encoder, RoPE, RMSNorm, SwiGLU, learned question/type embeddings, three heads, supervised training, temperature scaling, benchmark datasets and evaluation protocol.

The implementation must never be described as TypeSafe Jev's internal architecture or RLCD reproduction.
