# Experiments

## Fairness controls

1. Build the underlying records once and derive the same state/question/ground-truth examples for both models.
2. Split with one fixed seed into train/validation/test. The public test split is used as test data; no test tuning is performed.
3. Fit the tokenizer on training information only.
4. Use repeated seeds `42,43,44` by default.
5. Report mean and standard deviation from the saved per-seed runs when aggregating results externally.
6. Parameter-matched mode keeps the Mini-LLM configuration fixed and searches a separate Mini-Jev configuration with the closest parameter count.
7. Compute-matched mode uses identical batch size, optimizer-step count and number of training examples seen by both models.
8. Temperature scaling is fitted on validation logits only and then applied unchanged to the test set.

## Decision tasks

The common benchmark uses three typed decisions from the same state:

- Intent: Choice over the dataset labels.
- Routing: Choice over a deterministic operational routing taxonomy.
- Risk: Score over low/medium/high levels derived from explicit intent-name rules.

The derived routing/risk labels are benchmark design choices and must not be presented as human-annotated dataset labels.

The Mini-Jev forward pass computes one shared state representation and then evaluates all questions in the same forward call. A serial baseline repeatedly calls the model once per question. The benchmark records both latencies.

## Probability evaluation

For Mini-Jev, the head softmax directly supplies the probability distribution.

For Mini-LLM, option probabilities are estimated from normalized conditional log-likelihoods of the candidate options following the decision prompt. This creates a common probabilistic comparison without pretending that free-form generation is natively a categorical decision head.

Generated JSON/text is evaluated separately for structured-output failures.

## Calibration

Reported metrics include Brier score, ECE and NLL. Reliability-bin data and confidence histograms are saved. Temperature scaling is a post-training baseline only; it is not TypeSafe RLCD.

## Distribution shift

The OOD experiment applies a deterministic lexical transformation to held-out states and evaluates without adaptation. The comparison records accuracy and ECE degradation.

## Language generation

TinyStories is a separate next-token language-modeling experiment. Its loss and generation behavior are not merged with decision-task metrics.

## What is not measured automatically

A full benchmark requires the datasets to be present locally and enough compute/time to train all requested seeds. Until those commands are run, `results/` should be treated as unmeasured rather than as evidence.
