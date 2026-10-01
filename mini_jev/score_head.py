import torch, torch.nn as nn
class ScoreHead(nn.Module):
    def __init__(self,d,max_levels=10):
        super().__init__(); self.levels=nn.Parameter(torch.randn(max_levels,d)*0.02); self.net=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,d))
    def forward(self,h,q,n):
        base=self.net(torch.cat([h,q],-1)); return (self.levels[:n]*base).sum(-1)
