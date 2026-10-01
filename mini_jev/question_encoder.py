import torch, torch.nn as nn
class QuestionEncoder(nn.Module):
    TYPES={"choice":0,"score":1,"noul":2}
    def __init__(self,vocab_size,d_model): super().__init__(); self.emb=nn.Embedding(vocab_size,d_model); self.type_emb=nn.Embedding(3,d_model); self.proj=nn.Linear(d_model,d_model)
    def encode_tokens(self,ids):
        m=(ids!=0).unsqueeze(-1); h=self.emb(ids); return (h*m).sum(1)/m.sum(1).clamp_min(1)
    def forward(self,ids,qtype):
        t=torch.tensor([self.TYPES[qtype]],device=ids.device); return self.proj(self.encode_tokens(ids)+self.type_emb(t))
