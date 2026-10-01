import torch, torch.nn as nn, torch.nn.functional as F
from .rope import apply_rope
class CausalSelfAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.1):
        super().__init__(); assert d_model%n_heads==0; self.h=n_heads; self.d=d_model//n_heads; self.qkv=nn.Linear(d_model,3*d_model); self.out=nn.Linear(d_model,d_model); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        b,t,_=x.shape; q,k,v=self.qkv(x).chunk(3,-1); q=q.view(b,t,self.h,self.d).transpose(1,2); k=k.view(b,t,self.h,self.d).transpose(1,2); v=v.view(b,t,self.h,self.d).transpose(1,2); q,k=apply_rope(q,k); mask=torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),1); a=(q@k.transpose(-2,-1))/self.d**0.5; a=a.masked_fill(mask,-torch.inf); a=self.drop(F.softmax(a,-1)); return self.out((a@v).transpose(1,2).contiguous().view(b,t,-1))
