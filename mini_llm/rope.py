import torch

def rotate_half(x):
    a = x[..., ::2]
    b = x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)

def apply_rope(q, k, base=10000.0):
    d = q.size(-1)
    if d % 2:
        raise ValueError("RoPE head dimension must be even")
    pos = torch.arange(q.size(-2), device=q.device, dtype=q.dtype)
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=q.device, dtype=q.dtype) / d))
    angles = torch.outer(pos, inv)
    cos = angles.cos().repeat_interleave(2, dim=-1)[None, None]
    sin = angles.sin().repeat_interleave(2, dim=-1)[None, None]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
