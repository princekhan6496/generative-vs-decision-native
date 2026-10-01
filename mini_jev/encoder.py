import torch, torch.nn as nn
from .attention import EncoderSelfAttention
class EncoderBlock(nn.Module):
    def __init__(self,d,h,dff,dropout):
        super().__init__(); self.n1=nn.LayerNorm(d); self.attn=EncoderSelfAttention(d,h,dropout); self.n2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,dff),nn.SiLU(),nn.Linear(dff,d)); self.drop=nn.Dropout(dropout)
    def forward(self,x,pad):
        x=x+self.drop(self.attn(self.n1(x),pad)); return x+self.drop(self.ff(self.n2(x)))
class StateEncoder(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,n_layers,d_ff,dropout=.1,max_seq_len=128):
        super().__init__(); self.emb=nn.Embedding(vocab_size,d_model,padding_idx=0); self.pos=nn.Embedding(max_seq_len,d_model); self.blocks=nn.ModuleList([EncoderBlock(d_model,n_heads,d_ff,dropout) for _ in range(n_layers)]); self.norm=nn.LayerNorm(d_model)
    def forward(self,x):
        pos=torch.arange(x.size(1),device=x.device); h=self.emb(x)+self.pos(pos)[None,:,:]; pad=x.eq(0)
        for b in self.blocks: h=b(h,pad)
        h=self.norm(h); mask=(~pad).unsqueeze(-1); return (h*mask).sum(1)/mask.sum(1).clamp_min(1)
