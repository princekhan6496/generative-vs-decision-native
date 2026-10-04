# Generative LLM vs Decision-Native Model for AI Automation

## Research Project

This repository presents an educational PyTorch study comparing two ways of modeling AI-automation decisions:

1. **Mini-LLM** — a small decoder-only generative Transformer that represents decisions through text.
2. **Mini-Jev** — a small decision-native architecture with explicit Choice, Score and Noul outputs, inspired only by publicly documented TypeSafe/System One concepts.

> **Important:** Mini-Jev is an original educational implementation. It is **not a reproduction, reimplementation, reverse engineering, or claimed replica of TypeSafe Jev**. TypeSafe does not publicly disclose enough internal architectural or training information to support such a claim.

---

# 1. Research Question

Large language models are general-purpose sequence generators. They can represent many tasks through text, including decisions such as classification, routing and risk assessment.

A decision-native model takes a different approach: instead of generating a textual answer, it represents the decision explicitly as a typed probability distribution.

This project asks:

> **When the same structured decision problem is given to a small generative LLM and a small decision-native model under controlled conditions, how do they differ in decision quality, probabilistic quality, seed stability, structured-output reliability and measured computational cost?**

The project does **not** attempt to answer whether one architecture is universally better.

Instead, it studies the trade-offs between the two approaches.

---

# 2. Research Hypotheses

The experiments investigate the following hypotheses:

### H1 — Decision quality
A small generative LLM can solve fixed structured decision tasks by representing the decision as text, but a native decision head may provide a more direct formulation of the task.

### H2 — Probabilistic behavior
A decision-native architecture should have a direct probability distribution over its allowed decisions, while an LLM must derive comparable probabilities from candidate-option likelihoods.

### H3 — Stability
The two architectures may react differently to random initialization and training randomness. Repeated seeds are therefore required rather than relying on one run.

### H4 — Multi-question inference
A model that computes a state representation once and reuses it across multiple questions may have a systems-level advantage when many decisions refer to the same state.

### H5 — Generative flexibility
The Mini-LLM should remain naturally suited to open-ended token generation, whereas the Mini-Jev is intentionally optimized around predefined decision types.

These hypotheses are empirical questions. The results do not automatically prove the hypotheses.

---

# 3. What We Built

## Mini-LLM

A decoder-only Transformer implemented in PyTorch with:

- train-only tokenizer fitting
- token embeddings
- RoPE
- causal multi-head self-attention
- RMSNorm
- SwiGLU
- residual connections
- vocabulary / LM head
- next-token cross-entropy
- AdamW
- gradient clipping
- checkpoints
- autoregressive generation
- temperature sampling
- top-k sampling
- top-p sampling

For the structured decision benchmark, the Mini-LLM estimates option probabilities using normalized conditional log-likelihoods of candidate answers.

This is an evaluation mechanism; it is **not** a native Choice/Score/Noul head.

## Mini-Jev

An educational decision-native architecture containing:

- shared state encoder
- typed question representation
- Choice head
- Score head
- Noul head
- supervised mixed-task training
- shared-state multi-question inference
- probability/confidence utilities
- validation-only temperature scaling

The implementation is inspired by the **public behavior** described for TypeSafe/System One models. Its internal architecture is independently designed.

---

# 4. Experimental Design

The main decision benchmark uses BANKING77.

Both architectures receive the same underlying examples and labels.

The benchmark measures:

- accuracy
- macro F1
- precision
- recall
- Brier score
- ECE
- NLL
- structured-output failure rate
- parameter count
- training time
- optimizer steps
- examples seen
- tokens processed
- multi-question latency
- serial-vs-shared-state latency
- throughput
- seed-to-seed variation

The standard repeated-seed experiment uses:

```text
42, 43, 44
```

The parameter-matched protocol keeps the Mini-LLM configuration fixed and searches for a Mini-Jev configuration with a nearly identical parameter count.

The measured parameter counts were approximately:

```text
Mini-LLM: 403,104
Mini-Jev: 403,818
difference: 714 parameters (~0.18%)
```

This controls parameter count closely, but it does **not** make the architectures equal in FLOPs, memory access or runtime.

---

# 5. Decision Tasks

The common BANKING77 state is used to derive several structured decisions.

### Intent — Choice

The original BANKING77 intent label.

### Routing — Choice

A deterministic operational taxonomy:

```text
account
card
cash
payment
transfer
other
```

### Risk — Score

A three-level benchmark score:

```text
low
medium
high
```

### Security — Noul

A binary yes/no decision derived using explicit benchmark rules.

The routing, risk and security labels are **benchmark constructions**. They are not additional human annotations supplied by BANKING77.

---

# 6. Main Experimental Results

The following results come from the completed **parameter-matched BANKING77 runs with seeds 42, 43 and 44**.

## 6.1 Intent accuracy

| Seed | Mini-LLM | Mini-Jev |
|---|---:|---:|
| 42 | 23.73% | 17.86% |
| 43 | 13.64% | 16.92% |
| 44 | 18.64% | 16.49% |
| **Mean** | **18.67%** | **17.09%** |
| **Sample SD** | **5.05 pp** | **0.70 pp** |

The Mini-LLM has a slightly higher mean accuracy:

**18.67% vs 17.09% (+1.58 percentage points).**

However, the more interesting observation is the variation across seeds.

Mini-LLM:

```text
23.73 → 13.64 → 18.64%
```

Mini-Jev:

```text
17.86 → 16.92 → 16.49%
```

Thus, **Mini-Jev was considerably more stable across the three seeds in this experiment**, while Mini-LLM achieved a higher mean but showed much larger variation.

This is an observed result of these three runs, not a claim that decision-native models are inherently always more stable.

## 6.2 Macro F1

| Metric | Mini-LLM | Mini-Jev |
|---|---:|---:|
| Mean Macro F1 | **14.88%** | 12.58% |
| Sample SD | 4.34 pp | 0.34 pp |

Mini-LLM again had the higher mean, while Mini-Jev was substantially more consistent across seeds.

## 6.3 Probability quality

| Metric | Mini-LLM | Mini-Jev | Lower is better |
|---|---:|---:|---|
| Mean Brier | **0.915** | 0.925 | Yes |
| Mean NLL | 3.435 | **2.909** | Yes |

The results show a mixed trade-off:

- Mini-LLM had a slightly lower mean Brier score.
- Mini-Jev had substantially lower mean NLL.

Therefore, neither probability metric alone establishes a universal winner.

## 6.4 Noul accuracy

| Seed | Mini-LLM | Mini-Jev |
|---|---:|---:|
| 42 | 95.42% | 93.18% |
| 43 | 95.29% | 93.41% |
| 44 | 94.74% | 92.56% |
| **Mean** | **95.15%** | 93.05% |

The Noul task was easier than the 77-class intent task. Its accuracy should therefore not be directly compared with intent accuracy.

The positive class was also relatively uncommon, so accuracy alone is not sufficient to characterize the binary task.

## 6.5 Training cost

| Seed | Mini-LLM | Mini-Jev |
|---|---:|---:|
| 42 | 125.04 s | 1778.72 s |
| 43 | 123.48 s | 1637.59 s |
| 44 | 125.97 s | 1585.06 s |
| **Mean** | **124.83 s** | **1667.12 s** |

In this implementation and training procedure, Mini-Jev took substantially longer to train.

This is a **measurement of this implementation**, not evidence that TypeSafe Jev itself is slower than an LLM.

The protocols also do not provide exact FLOP matching.

## 6.6 Structured output

The completed parameter-matched runs showed essentially zero structured-output failures.

This means the Mini-LLM was able to produce a parseable/mappable decision format in almost all evaluated cases.

This result is important because free-form generation can theoretically produce malformed outputs, but in these runs that failure mode was not a major contributor to the final result.

---

# 7. What We Learned

The central result is **not simply “LLM wins” or “Jev wins.”**

The experiment revealed a trade-off.

### Finding 1 — Accuracy was close

The average intent accuracy difference was relatively small:

```text
Mini-LLM   18.67%
Mini-Jev   17.09%
```

Mini-LLM was ahead by 1.58 percentage points.

### Finding 2 — Stability was very different

Mini-Jev's accuracy stayed in a narrow range:

```text
16.49% – 17.86%
```

Mini-LLM varied much more:

```text
13.64% – 23.73%
```

Therefore, **seed stability is one of the strongest observations from the experiment**.

### Finding 3 — Probabilistic quality was mixed

Mini-LLM had slightly better Brier score, while Mini-Jev had better NLL.

This suggests that looking only at classification accuracy would hide important differences in how the models represent uncertainty.

### Finding 4 — Mini-Jev was expensive to train in this implementation

Despite nearly identical parameter counts, Mini-Jev required substantially more measured training time under the implemented benchmark.

This demonstrates why parameter count alone is not an adequate measure of computational cost.

### Finding 5 — The two models solve the task differently

The Mini-LLM asks:

> “What tokens should I generate to express the answer?”

The Mini-Jev asks:

> “Given this state and typed question, what probability distribution over the allowed decisions should I output?”

This is the central architectural distinction being investigated.

---

# 8. Important Interpretation of the Low Accuracy

The absolute BANKING77 intent accuracy is low for both educational models.

This should be acknowledged rather than hidden.

The project is **not a state-of-the-art BANKING77 classifier**.

The experiment uses small models and is designed primarily to study architectural trade-offs under a controlled parameter budget.

Therefore:

> **18.67% accuracy should not be presented as strong BANKING77 classification performance.**

Instead, the scientifically useful observations are the relative behavior between the two implementations:

- mean task performance
- probability quality
- seed stability
- structured-output reliability
- measured training cost
- multi-question inference behavior

A stronger absolute classifier could require a larger model, more training, hyperparameter optimization and task-specific improvements. Such improvements would be a separate engineering objective from the architectural comparison presented here.

---

# 9. Overall Conclusion

> **Under the parameter-matched BANKING77 experiment, the Mini-LLM achieved slightly higher mean intent accuracy and macro F1, while Mini-Jev achieved substantially more stable intent accuracy across random seeds and lower mean NLL. The Mini-LLM also required much less measured training time in this implementation. These results therefore indicate a trade-off rather than a universal winner: generative modeling can remain competitive for fixed structured decisions, while a decision-native formulation provides an explicit decision interface and showed more stable behavior across the tested seeds.**

The project does **not** conclude that Mini-Jev is better than LLMs or that LLMs are better than decision-native models.

It concludes that **the representation of a decision — generated text versus an explicit typed decision distribution — changes the measurable behavior of a small model across accuracy, uncertainty, stability and computational cost.**

---

# 10. Limitations

1. Both models are educational-scale implementations.
2. BANKING77 intent accuracy is low compared with specialized large classifiers.
3. Only three random seeds were used in the main repeated-seed comparison.
4. The parameter-matched protocol controls parameter count, not exact compute.
5. The compute-matched protocol controls examples, batch size and optimizer steps, not exact FLOPs.
6. Derived routing/risk/security tasks are benchmark constructions, not additional human labels.
7. Latency and training time depend on implementation and hardware.
8. The Mini-Jev architecture is not TypeSafe Jev's disclosed architecture.
9. The RLCD-inspired experiment is not TypeSafe RLCD.
10. Results from this repository cannot be used as measurements of TypeSafe's production Jev.

---

# 11. TypeSafe Boundary

TypeSafe publicly describes System One models around typed probabilistic decisions, including Noul, Choice and Score, multiple questions over shared state, a new architecture, parallel sampling and RLCD.

The public material does not disclose sufficient internal details to reproduce Jev faithfully.

Therefore, the correct description is:

> **“Mini-Jev, an educational decision-native architecture inspired by publicly documented Jev/System One concepts.”**

Not:

> “A reproduction of Jev.”

See `VERIFIED_VS_ASSUMED.md` for the detailed source boundary.

---

# 12. Reproducibility

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install:

```bash
pip install -r requirements.txt
```

Verify:

```bash
pytest -q
python scripts/smoke_test.py
python scripts/tiny_train.py
```

Download supported data:

```bash
python scripts/download_data.py
```

Run parameter-matched BANKING77:

```bash
python experiments/benchmark.py --dataset banking77 --banking-train data/banking77_train.csv --banking-test data/banking77_test.csv --protocol parameter-matched --seeds 42,43,44 --out results/banking77_parameter_matched.json
```

Run compute-matched:

```bash
python experiments/benchmark.py --dataset banking77 --banking-train data/banking77_train.csv --banking-test data/banking77_test.csv --protocol compute-matched --seeds 42,43,44 --out results/banking77_compute_matched.json
```

Run CLINC150:

```bash
python experiments/benchmark.py --dataset clinc150 --clinc-json data/clinc150.json --epochs 3 --seeds 42,43,44 --out results/clinc150.json
```

Run lexical distribution shift:

```bash
python experiments/ood.py --banking-train data/banking77_train.csv --banking-test data/banking77_test.csv --epochs 3 --out results/ood.json
```

Run TinyStories separately:

```bash
python experiments/tiny_stories.py --text data/tinystories.txt --epochs 3 --out results/tinystories.json
```

---

# 13. Research Integrity

The repository distinguishes:

- public TypeSafe facts
- project architecture choices
- measured results
- benchmark constructions
- limitations

No result should be described as measured until the corresponding experiment has actually been executed.

Unrun experiments are **unmeasured**, not estimated or fabricated.

## References

- TypeSafe: https://typesafe.ai/
- TypeSafe announcement: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe evaluations: https://evals.typesafe.ai/
- BANKING77: https://huggingface.co/datasets/PolyAI/banking77
- CLINC OOS: https://github.com/clinc/oos-eval
- TinyStories: https://huggingface.co/datasets/roneneldan/TinyStories
