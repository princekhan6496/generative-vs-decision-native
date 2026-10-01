import torch

def rlcd_inspired_objective(logits, targets, confidence_weight=0.1):
    """Educational calibrated-decision objective; NOT TypeSafe RLCD."""
    ce=torch.nn.functional.cross_entropy(logits,targets)
    p=torch.softmax(logits,-1); conf=p.max(-1).values; correct=(p.argmax(-1)==targets).float()
    calibration=(conf-correct).pow(2).mean()
    return ce+confidence_weight*calibration

# This module is deliberately opt-in. It must never be described as TypeSafe's RLCD.
