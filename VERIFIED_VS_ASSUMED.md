# Verified vs Assumed

This document separates what this project can verify from TypeSafe's public material from what is independently designed and implemented in this repository.

## Verified from public TypeSafe material

The following points are supported by TypeSafe's public announcement, website, API documentation and workflow-evaluation material:

- **System One Models** are presented by TypeSafe as a model class designed for decisions inside software and automation.
- **Jev** is presented as TypeSafe's first public System One model.
- TypeSafe describes Jev as taking state plus structured questions and returning **typed decisions** with probabilities and confidence.
- Public TypeSafe material documents three decision primitives used in its workflow evaluations:
  - **Noul** — yes/no decision.
  - **Choice** — one option from a defined set, with a probability distribution and confidence.
  - **Score** — a rating on a scale, with a score/distribution and confidence.
- TypeSafe describes multiple questions being evaluated against the same state and emphasizes parallel evaluation/sampling.
- TypeSafe describes Jev as a model that gives up general string generation in favor of structured decision outputs.
- TypeSafe states that Jev uses a **new model architecture**, a **parallel sampler**, and a training method called **Reinforcement Learning for Calibrated Decisions (RLCD)**.
- TypeSafe's public materials do not disclose enough internal implementation detail to establish the exact Transformer structure, layer count, hidden size, attention design, tokenizer, weights, optimization schedule, sampler implementation or RLCD algorithm used internally by Jev.
- The public API documentation confirms a System One endpoint in which one state can be accompanied by one or more questions, and the questions can use the documented decision types.

These public descriptions establish the **behavioral inspiration** for this project. They do not establish the internal architecture of Jev.

## Our implementation choices

The following are choices made by this repository and must not be presented as facts about Jev:

- Transformer encoder for the Mini-Jev shared state representation.
- Token embeddings and positional representations in the Mini-Jev encoder.
- RoPE in the Mini-LLM.
- RMSNorm.
- SwiGLU.
- Specific attention dimensions, layer counts and hidden sizes.
- Separate Mini-Jev question encoder.
- Learned question/type representations.
- Separate Choice, Score and Noul prediction heads.
- Supervised mixed-task training.
- Cross-entropy and binary cross-entropy objectives used by the educational model.
- AdamW and gradient clipping settings.
- Temperature scaling as a calibration baseline.
- BANKING77 and CLINC150 as benchmark datasets.
- The derived routing and risk tasks.
- Conditional option log-likelihoods as the Mini-LLM probability comparison method.
- The fixed lexical distribution-shift transformation.
- The parameter-matching search procedure.
- The `compute-matched` workload-control protocol.
- The serial-versus-shared-state latency experiment.
- The TinyStories language-generation experiment.
- Any optional RLCD-inspired educational objective in this repository.

## What the project does not claim

This project does **not** claim:

- to reproduce TypeSafe Jev;
- to reimplement TypeSafe's proprietary architecture;
- to reverse engineer Jev;
- to match Jev's internal parameter count;
- to match Jev's hidden size, number of layers, attention mechanism or tokenizer;
- to reproduce TypeSafe's sampler;
- to reproduce TypeSafe's training data;
- to reproduce TypeSafe's RLCD algorithm;
- to reproduce TypeSafe's production inference stack;
- that the Mini-Jev is equivalent to Jev;
- that measurements from this repository are measurements of TypeSafe's model.

## Correct terminology

Preferred wording:

> **“Mini-Jev, an educational decision-native architecture inspired by publicly documented Jev/System One concepts.”**

Also acceptable:

> **“A decision-native model implemented for educational research, inspired by the public behavioral description of TypeSafe's Jev.”**

Avoid:

> “reimplementation of Jev”

> “reproduction of Jev”

> “reverse-engineered Jev”

> “our version of TypeSafe's architecture”

> “Jev's architecture is ...”

unless the statement is directly supported by a public TypeSafe source.

## Public facts vs project behavior

A useful distinction is:

**Public TypeSafe behavior**

State + typed questions → typed probabilistic decisions.

**This repository**

State → educational Transformer representation + question representation → Choice / Score / Noul heads → probabilities and decisions.

The second line is an implementation of the research idea, not a description of Jev's hidden internals.

## RLCD wording

TypeSafe publicly names its training method **Reinforcement Learning for Calibrated Decisions (RLCD)**.

If this repository runs `mini_jev/rlcd_experiment.py`, describe it as:

> **“an RLCD-inspired educational objective”**

It must not be described as TypeSafe RLCD, an implementation of TypeSafe RLCD, or a reproduction of the Jev training algorithm.

Likewise, temperature scaling in the main benchmark is simply a **post-training calibration baseline**. It is not RLCD.

## Source boundary

Primary TypeSafe sources:

- https://typesafe.ai/
- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://api.typesafe.ai/docs
- https://api.typesafe.ai/redoc
- https://evals.typesafe.ai/

The benchmark datasets and their licenses are separate from TypeSafe and must be treated according to their own terms.
