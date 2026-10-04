# Experiments

This document defines the experimental protocol used to compare the educational Mini-LLM and Mini-Jev implementations.

## 1. Research questions

The experiments are intended to study:

1. Whether a small decoder-only generative model can solve fixed structured decision tasks when decisions are represented through text.
2. Whether a decision-native model is effective when the output space is explicitly typed.
3. How the two approaches compare in predictive accuracy and macro F1.
4. How their probability estimates compare using Brier score, ECE and NLL.
5. How stable the results are across repeated random seeds.
6. What training and inference costs are measured for these implementations.
7. Whether evaluating several questions from one shared state provides a measurable latency benefit.
8. How performance changes under the selected lexical distribution shift.
9. How the generative model behaves on a separate open-ended language-generation task.

These questions are about the repository's implementations, not about the undisclosed internals of TypeSafe Jev.

## 2. Fairness controls

### 2.1 Common examples

The underlying records are constructed once and then converted into the same state/question/ground-truth examples for both architectures.

Neither architecture receives a different set of examples for the decision benchmark.

### 2.2 Data split

A deterministic split is used to create training and validation data from the training distribution. The public test split is kept for held-out evaluation.

The test set is not used for model fitting or temperature-scaling fitting.

### 2.3 Tokenizer fitting

The Mini-LLM tokenizer is fitted using training information only.

Test information must not influence tokenizer fitting.

### 2.4 Repeated seeds

The standard repeated-seed protocol uses:

```text
42, 43, 44
```

Per-seed results should be saved before calculating an aggregate.

For an aggregate report, calculate the mean and standard deviation from the saved per-seed results rather than manually replacing missing runs with estimates.

### 2.5 Parameter matching

In `parameter-matched` mode:

- the Mini-LLM configuration is kept fixed;
- a separate Mini-Jev configuration is searched;
- the goal is to make parameter counts as close as practical.

This controls approximate model size.

It does **not** make the architectures equivalent in FLOPs, memory access, sequence processing or runtime.

### 2.6 Workload matching

In `compute-matched` mode, the current benchmark controls:

- batch size;
- optimizer-step count;
- number of training examples seen.

The benchmark does **not** calculate or equalize exact FLOPs.

Therefore the precise description is:

> **controlled training-workload matching, not exact FLOP matching.**

The benchmark also reports tokens processed and measured training time so that computational differences are visible.

### 2.7 Calibration fitting

Temperature scaling is fitted on validation outputs only:

```text
p = softmax(z / T)
```

The fitted temperature is then frozen and applied to test outputs.

The test set is never used to choose the temperature.

Temperature scaling is a calibration baseline and must not be described as TypeSafe RLCD.

## 3. Decision tasks

The common BANKING77 benchmark uses three typed decision tasks derived from the same state.

### Intent

A **Choice** over the dataset's intent labels.

The input state is the BANKING77 utterance, and the target is its original dataset category.

### Routing

A **Choice** over a smaller operational taxonomy:

```text
account
card
cash
payment
transfer
other
```

The routing target is deterministically derived from the intent/category.

### Risk

A **Score** over:

```text
low
medium
high
```

The risk target is deterministically derived from explicit intent-name rules defined by the benchmark.

### Security Noul

A **Noul** yes/no decision is also used by the implementation.

Its positive/negative target is derived by explicit benchmark rules rather than supplied as an additional human annotation.

### Important labeling rule

The derived routing, risk and security labels are **benchmark constructions**.

They must not be presented as human-annotated BANKING77 ground truth.

## 4. Mini-Jev inference protocol

The Mini-Jev computes one shared state representation and reuses it for the questions in the same inference call.

Conceptually:

```text
state
  ↓
shared state encoder
  ↓
shared representation
  ├── question 1 → typed head
  ├── question 2 → typed head
  ├── question 3 → typed head
  └── question 4 → typed head
```

The benchmark records:

- one multi-question / shared-state call;
- a serial baseline that invokes the model separately for each question.

The serial baseline is useful for measuring the effect of shared-state execution in this implementation.

A measured speedup is **not guaranteed**. Results can depend on batch size, sequence lengths, hardware, implementation overhead and the number of questions.

## 5. Probability evaluation

### Mini-Jev

For Choice and Score tasks, the head directly produces a probability distribution.

For Noul, the model produces a probability through its binary output.

### Mini-LLM

The Mini-LLM does not have a native Choice/Score/Noul interface.

For comparable probability metrics, the benchmark estimates option probabilities using normalized conditional log-likelihoods of candidate options following the decision prompt.

This gives a common categorical probability representation for evaluation.

It must not be described as a native decision head.

### Structured generation

Generated structured output is evaluated separately.

The benchmark records whether the generated output can be parsed and mapped to the expected decision.

A malformed or unmappable generated answer counts as a structured-output failure.

Probability quality and generated-format reliability are therefore separate measurements.

## 6. Metrics

### Classification quality

- **Accuracy:** fraction of correct predictions.
- **Macro F1:** F1 averaged equally across classes.
- **Precision:** precision of predicted classes.
- **Recall:** recall of predicted classes.

Macro F1 is particularly useful when class frequencies are uneven.

### Probability quality

- **Brier score:** squared probabilistic error; lower is better.
- **NLL:** negative log-likelihood of the target; lower is better.
- **ECE:** expected calibration error; lower generally indicates closer agreement between confidence and empirical accuracy.

Calibration results should be interpreted together with the underlying accuracy and class distribution.

### Reliability data

Reliability-bin data and confidence histograms are saved so calibration can be inspected instead of relying only on a single scalar.

### Systems measurements

The benchmark records:

- parameter count;
- measured training time;
- optimizer steps;
- examples seen;
- tokens processed where applicable;
- latency;
- throughput;
- structured-output failure rate.

Measurements are implementation- and hardware-dependent.

## 7. Distribution shift

The OOD experiment applies a fixed lexical transformation to held-out test states.

The transformation is chosen before evaluation and is not learned from the shifted test set.

No tuning or calibration is performed on the shifted test data.

The experiment compares performance and calibration degradation between the original and transformed inputs.

This is a controlled lexical-shift experiment, not a claim that it represents all real-world distribution shifts.

## 8. Language generation

TinyStories is evaluated separately as a next-token language-modeling task.

Its metrics measure language-generation behavior, not decision quality.

Therefore:

- TinyStories loss must not be averaged with decision-task Brier/F1/accuracy.
- TinyStories generation results must not be used to claim superiority on structured automation tasks.
- Decision-task results must not be used to claim general language-model superiority.

## 9. Reporting repeated seeds

For each metric:

```text
mean = average of the saved per-seed values
standard deviation = sample standard deviation across seeds
```

Report the individual seed values as well as the aggregate where practical.

Do not report an aggregate if required seed runs are missing without clearly labeling it as incomplete.

## 10. What the experiments cannot establish

These experiments cannot establish:

- the internal architecture of TypeSafe Jev;
- that Mini-Jev is equivalent to Jev;
- that TypeSafe uses the same Transformer components implemented here;
- that the measured training-time ratio would hold for TypeSafe's production system;
- that the measured latency represents TypeSafe's service;
- that the educational RLCD-inspired objective reproduces RLCD;
- that the benchmark's derived routing/risk labels represent human judgments;
- that one architecture is universally better for all AI automation workloads.

## 11. What is not measured automatically

A complete repeated-seed benchmark requires:

- the required datasets to be present locally;
- the requested experiments to actually be run;
- sufficient compute and time for all seeds.

Until an experiment has been executed, its result should be described as:

> **unmeasured**

It must not be described as zero, estimated, expected, or completed.

## 12. Recommended interpretation

The strongest conclusions should be narrow and evidence-based.

For example:

> “Under this parameter-matched protocol, Mini-LLM and Mini-Jev showed different trade-offs in task accuracy, probabilistic quality, stability and measured runtime.”

Avoid:

> “Mini-Jev is better than LLMs.”

or:

> “This proves Jev is faster.”

The experiments compare two educational implementations under a specified protocol. They do not benchmark TypeSafe's proprietary production model.
