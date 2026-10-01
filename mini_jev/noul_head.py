import torch.nn as nn
class NoulHead(nn.Module):
    def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,1))
    def forward(self,h,q): return self.net(__import__('torch').cat([h,q],-1)).squeeze(-1)
