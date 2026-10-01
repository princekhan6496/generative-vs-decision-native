# Verified vs Assumed

## Verified from public TypeSafe material

- Jev is presented as TypeSafe's first public System One model.
- System One models are described as decision-oriented models for software/automation.
- Inputs can include state plus typed questions.
- Public examples use Noul, Choice and Score decision primitives.
- Structured decisions include probabilities and confidence information.
- TypeSafe describes parallel sampling/evaluation rather than ordinary autoregressive string generation.
- TypeSafe says Jev uses a new model architecture and a training method called Reinforcement Learning for Calibrated Decisions (RLCD).
- The public announcement does not disclose enough internal implementation detail to reproduce the proprietary architecture or RLCD algorithm.

## Our implementation choices

- Transformer encoder for the shared state representation.
- RoPE, RMSNorm and SwiGLU.
- Separate question encoder and typed heads.
- Supervised Choice/Score/Noul losses.
- Temperature scaling as a calibration baseline.
- BANKING77 and CLINC150 benchmark construction.
- Derived routing/risk tasks.
- Conditional option likelihoods for the Mini-LLM probability comparison.
- Fixed lexical OOD transformation.
- Parameter/compute matching protocols.

## Important wording rule

Use: **"Mini-Jev, an educational decision-native architecture inspired by publicly documented Jev/System One concepts."**

Do not use: **"reimplementation," "reproduction," "reverse engineered Jev,"** or claims that any internal layer, parameter count, attention design, sampler, weights or RLCD algorithm matches TypeSafe.
