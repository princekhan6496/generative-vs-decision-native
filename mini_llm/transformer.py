import torch, torch.nn as nn
from .attention import CausalSelfAttention
class RMSNorm(nn.Module):
    def __init__(self,d,eps=1e-6): super().__init__(); self.w=nn.Parameter(__import__('torch').ones(d)); self.eps=eps
    def forward(self,x): return x*__import__('torch').rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.w
class SwiGLU(nn.Module):
    def __init__(self,d,dff): super().__init__(); self.w1=nn.Linear(d,dff); self.w2=nn.Linear(d,dff); self.w3=nn.Linear(dff,d)
    def forward(self,x): return self.w3(torch.nn.functional.silu(self.w1(x))*self.w2(x))
class DecoderBlock(nn.Module):
    def __init__(self,d,h,dff,dropout): super().__init__(); self.n1=RMSNorm(d); self.attn=CausalSelfAttention(d,h,dropout); self.n2=RMSNorm(d); self.ff=SwiGLU(d,dff); self.drop=nn.Dropout(dropout)
    def forward(self,x): x=x+self.drop(self.attn(self.n1(x))); return x+self.drop(self.ff(self.n2(x)))
class DecoderTransformer(nn.Module):
    def __init__(self,d,h,l,dff,dropout): super().__init__(); self.blocks=nn.ModuleList([DecoderBlock(d,h,dff,dropout) for _ in range(l)]); self.norm=RMSNorm(d)
    def forward(self,x):
        for b in self.blocks: x=b(x)
        return self.norm(x)
