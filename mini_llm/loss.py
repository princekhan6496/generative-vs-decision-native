import torch.nn.functional as F

def causal_lm_loss(logits, targets, pad_id=0):
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        targets[:, 1:].reshape(-1),
        ignore_index=pad_id,
    )
