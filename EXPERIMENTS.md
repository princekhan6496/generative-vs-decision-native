# Experiments and Research Findings

## Abstract

This study compares a small decoder-only generative language model (**Mini-LLM**) with an educational decision-native model (**Mini-Jev**) for structured AI-automation decisions. Mini-Jev is inspired by publicly documented TypeSafe/System One concepts but is not a reproduction of TypeSafe Jev.

The study evaluates decision accuracy, macro F1, probabilistic quality, seed stability, structured-output reliability, training cost and shared-state multi-question inference. The main BANKING77 experiment uses a parameter-matched protocol with three random seeds: 42, 43 and 44.

The measured results show a trade-off. Mini-LLM achieved higher mean intent accuracy (18.67% vs 17.09%) and macro F1 (14.88% vs 12.58%), while Mini-Jev achieved lower mean NLL (2.909 vs 3.435) and substantially lower seed-to-seed accuracy variation (0.70 vs 5.05 percentage points sample SD). Mini-LLM also trained much faster in this implementation. These results do not establish a universal winner; they demonstrate different behavior under the tested architecture and training conditions.

---

# 1. Research Objective

The objective is to experimentally compare two representations of an AI decision:

### Generative representation

```text
state → decoder Transformer → generated text
```

### Decision-native representation

```text
state + typed question → shared representation → typed probability distribution
```

The research question is:

> **Under controlled conditions, how do a small generative LLM and a small decision-native architecture differ in decision quality, uncertainty estimation, stability and computational behavior on structured AI-automation tasks?**

The study deliberately avoids treating TypeSafe Jev as an open implementation target because its internal architecture and training procedure are not publicly specified in sufficient detail.

---

# 2. Experimental Hypotheses

### H1 — Structured decision quality

A small generative LLM can perform fixed decision tasks by generating textual representations of decisions.

### H2 — Explicit probability modeling

A decision-native model provides probabilities directly through typed heads, whereas the Mini-LLM requires candidate-option likelihoods to construct a comparable probability distribution.

### H3 — Training stability

The architectures may respond differently to random initialization and optimization randomness.

### H4 — Shared-state inference

Reusing a state representation across multiple typed questions may change the latency characteristics of multi-question inference.

### H5 — Task specialization

A decision-native architecture may provide a more explicit interface for fixed decisions, while the generative architecture retains the ability to perform open-ended language generation.

These are experimental hypotheses, not assumptions that the results must confirm.

---

# 3. Common Benchmark

## 3.1 Dataset

The primary decision benchmark is BANKING77.

Each example contains a natural-language banking utterance and its intent category.

## 3.2 Common state

Both models receive the same underlying state.

## 3.3 Typed questions

The state is evaluated through:

- Intent — Choice
- Routing — Choice
- Risk — Score
- Security — Noul

Routing, risk and security targets are derived by explicit benchmark rules. They are not additional human annotations from BANKING77.

---

# 4. Fairness Controls

## 4.1 Same examples

The underlying records are constructed once and converted into common state/question/ground-truth examples.

Both architectures train and evaluate on the same decision examples.

## 4.2 Data separation

Training and validation data are created from the training distribution.

The public test split remains held out.

No test examples are used for model fitting or calibration fitting.

## 4.3 Tokenizer

The Mini-LLM tokenizer is fitted on training information only.

## 4.4 Repeated seeds

The main repeated-seed study uses:

```text
42, 43, 44
```

Per-seed results are retained and then aggregated.

## 4.5 Parameter matching

The Mini-LLM configuration is fixed.

A separate Mini-Jev configuration is searched to minimize parameter-count difference.

The completed parameter-matched experiments used approximately:

```text
Mini-LLM = 403,104 parameters
Mini-Jev = 403,818 parameters
difference = 714 parameters ≈ 0.18%
```

This is a close parameter-count match, not an exact compute match.

## 4.6 Workload matching

The `compute-matched` protocol matches:

- batch size
- optimizer-step count
- number of training examples

It does not calculate or equalize exact FLOPs.

Actual tokens and measured runtime are reported.

---

# 5. Probability Evaluation

## Mini-Jev

Choice and Score heads directly produce probability distributions.

Noul produces a binary probability.

## Mini-LLM

The Mini-LLM is a generative model and does not possess a native Choice/Score/Noul interface.

For comparable probability metrics, the benchmark:

1. constructs a decision prompt;
2. lists candidate options;
3. calculates normalized conditional log-likelihoods for candidate answers;
4. converts them into a probability distribution.

Generated structured output is evaluated separately.

This separation is important:

```text
probability evaluation ≠ generated-format evaluation
```

---

# 6. Metrics

### Decision quality

- Accuracy
- Macro F1
- Precision
- Recall

### Probability quality

- Brier score
- ECE
- NLL

### Systems behavior

- parameter count
- training time
- optimizer steps
- examples seen
- tokens processed
- inference latency
- throughput
- structured-output failure rate

### Stability

Mean and sample standard deviation are calculated across the three repeated seeds.

---

# 7. Main Results

## 7.1 Intent Accuracy

| Seed | Mini-LLM | Mini-Jev |
|---|---:|---:|
| 42 | 23.73% | 17.86% |
| 43 | 13.64% | 16.92% |
| 44 | 18.64% | 16.49% |
| **Mean** | **18.67%** | **17.09%** |
| **Sample SD** | **5.05 pp** | **0.70 pp** |

### Observation

Mini-LLM achieved the higher mean accuracy by:

```text
18.67% − 17.09% = 1.58 percentage points
```

However, Mini-LLM varied substantially across seeds.

Mini-LLM:

```text
23.73%
13.64%
18.64%
```

Mini-Jev:

```text
17.86%
16.92%
16.49%
```

The Mini-Jev standard deviation was much smaller.

### Research interpretation

The result indicates:

> **Mini-LLM had higher average intent accuracy, but Mini-Jev showed much greater seed stability in this experiment.**

This is one of the main findings of the study.

It should not be generalized beyond the tested models and three seeds.

---

# 8. Macro F1

| Metric | Mini-LLM | Mini-Jev |
|---|---:|---:|
| Mean | **14.88%** | 12.58% |
| Sample SD | 4.34 pp | 0.34 pp |

Mini-LLM again had higher mean performance.

Mini-Jev again showed lower variation across seeds.

This supports the same qualitative observation seen with intent accuracy:

> **The Mini-LLM reached a higher average score, while Mini-Jev was more consistent.**

---

# 9. Probability Quality

| Metric | Mini-LLM | Mini-Jev |
|---|---:|---:|
| Mean Brier | **0.915** | 0.925 |
| Mean NLL | 3.435 | **2.909** |

Lower is better for both metrics.

The results are therefore mixed:

- Brier favors Mini-LLM slightly.
- NLL favors Mini-Jev substantially.

This means the probability comparison cannot be reduced to one universal winner.

It demonstrates why structured-decision evaluation should include both task accuracy and uncertainty metrics.

---

# 10. Noul Results

| Seed | Mini-LLM | Mini-Jev |
|---|---:|---:|
| 42 | 95.42% | 93.18% |
| 43 | 95.29% | 93.41% |
| 44 | 94.74% | 92.56% |
| **Mean** | **95.15%** | 93.05% |

The Noul task is binary and therefore easier to solve than 77-class intent prediction.

Because its class distribution is imbalanced, accuracy alone should not be used as the only measure of binary decision quality.

---

# 11. Training Cost

| Seed | Mini-LLM | Mini-Jev |
|---|---:|---:|
| 42 | 125.04 s | 1778.72 s |
| 43 | 123.48 s | 1637.59 s |
| 44 | 125.97 s | 1585.06 s |
| **Mean** | **124.83 s** | **1667.12 s** |

Mini-Jev required substantially more measured training time in this implementation.

This observation must be worded carefully:

> **“Mini-Jev was substantially slower to train in this implementation and protocol.”**

It must not be written as:

> “Jev is slower than an LLM.”

The experiment does not measure TypeSafe's production system.

---

# 12. Structured Output Reliability

The completed parameter-matched runs produced essentially zero structured-output failures.

Therefore, in these experiments, the Mini-LLM's low intent accuracy cannot simply be explained by malformed generated output.

The model generally produced outputs that could be parsed/mapped, even though the predicted intent was often incorrect.

This is useful because it separates:

```text
format failure
```

from:

```text
decision error
```

---

# 13. Seed Stability as a Research Finding

This deserves separate treatment because it is one of the clearest differences in the experiment.

Mini-LLM intent accuracy range:

```text
23.73 − 13.64 = 10.09 percentage points
```

Mini-Jev intent accuracy range:

```text
17.86 − 16.49 = 1.37 percentage points
```

Therefore:

> **The Mini-Jev results were much less sensitive to the three tested random seeds, while Mini-LLM results changed considerably between runs.**

Possible explanations include differences in optimization dynamics and the difficulty of learning a multi-task structured-output behavior through a small generative model.

However, this study does not isolate the causal mechanism. More seeds and controlled ablations would be required to establish why the difference occurs.

---

# 14. The Low Absolute Accuracy: Interpretation and Limitation

The intent accuracy of both models is low.

This is an important limitation and should be reported openly.

The study is not intended to be a state-of-the-art BANKING77 classification benchmark.

The models are educational-scale implementations, and the study focuses on architectural behavior under a controlled parameter budget.

Therefore the correct claim is:

> **The experiment compares the relative behavior of two small architectures; it does not demonstrate competitive absolute BANKING77 classification performance.**

A stronger classifier could require:

- larger model capacity
- longer training
- hyperparameter tuning
- task-specific optimization
- more extensive training experiments

Those improvements would answer a different question.

The low accuracy should therefore be treated as a limitation, not hidden as a positive result.

---

# 15. Research Findings

### Finding 1 — Similar low absolute accuracy

Both architectures had low absolute intent accuracy in the current educational setup.

### Finding 2 — Mini-LLM had a slightly higher mean

Mini-LLM:

```text
18.67%
```

Mini-Jev:

```text
17.09%
```

### Finding 3 — Mini-Jev was substantially more stable

Mini-Jev had:

```text
0.70 pp sample SD
```

versus:

```text
5.05 pp sample SD
```

for Mini-LLM.

### Finding 4 — Probability metrics disagreed

Mini-LLM had slightly better Brier score.

Mini-Jev had better NLL.

### Finding 5 — Training cost differed strongly

Mini-Jev took substantially longer to train in the current implementation despite nearly equal parameter counts.

### Finding 6 — Structured output was not the primary failure mode

Mini-LLM structured-output failures were essentially zero in the final parameter-matched runs.

### Finding 7 — Parameter count alone does not explain behavior

The models had almost identical parameter counts but showed different accuracy stability, probability behavior and training time.

---

# 16. Overall Conclusion

The experiments do not support a universal “generative LLM vs decision-native model” winner.

Instead, they show a measurable trade-off:

```text
Mini-LLM
↑ slightly higher mean accuracy/F1
↑ slightly better mean Brier
↑ much lower measured training time
↓ much larger seed-to-seed variation
↓ probability behavior represented indirectly for decision evaluation

Mini-Jev
↑ much more stable accuracy across tested seeds
↑ better mean NLL
↑ native typed decision representation
↓ lower mean intent accuracy in this experiment
↓ substantially higher measured training cost
```

The central conclusion is:

> **For the tested small models, representing a decision as generated text and representing it as an explicit typed decision are not equivalent engineering choices. They produce different trade-offs in accuracy, uncertainty estimation, stability and computational cost.**

The study therefore supports **workload-dependent architecture selection**, not a universal claim that one paradigm replaces the other.

---

# 17. Limitations

1. Only three seeds were used.
2. Both models are small educational implementations.
3. Absolute BANKING77 accuracy is low.
4. Parameter matching does not imply exact FLOP matching.
5. `compute-matched` is workload matching, not exact compute matching.
6. Training time depends on implementation and hardware.
7. Derived routing/risk/security labels are not human annotations.
8. Mini-Jev does not reproduce TypeSafe's proprietary architecture.
9. The RLCD-inspired objective is not TypeSafe RLCD.
10. The experiment cannot establish how TypeSafe Jev itself would perform under this benchmark.

---

# 18. Further Experiments

A stronger research extension would include:

- more random seeds;
- larger model sizes;
- training curves;
- ablation of the Mini-Jev shared encoder;
- ablation of typed heads;
- more decision datasets;
- exact FLOP accounting;
- controlled inference batch-size experiments;
- stronger OOD transformations;
- additional calibration methods;
- larger language-generation evaluation.

These are future experiments, not current results.

---

# 19. Reproducibility Rule

Every reported number must come from an executed experiment.

If an experiment has not been run:

```text
status = unmeasured
```

It must not be represented as an expected, estimated or fabricated result.
