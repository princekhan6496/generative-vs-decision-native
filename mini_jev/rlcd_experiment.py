"""Educational RLCD-inspired objective. NOT TypeSafe RLCD."""
import torch
def rlcd_inspired_objective(logits,targets):
    """A simple proper-scoring objective; this is not a reconstruction of TypeSafe RLCD."""
    return torch.nn.functional.cross_entropy(logits,targets)
