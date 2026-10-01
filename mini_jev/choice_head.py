import torch, torch.nn as nn
class ChoiceHead(nn.Module):
    def __init__(self,d): super().__init__(); self.q=nn.Linear(d,d); self.o=nn.Linear(d,d); self.bias=nn.Linear(d,1)
    def forward(self,h,option_reps):
        q=self.q(h); opts=torch.stack(option_reps); return (opts@q)+self.bias(h).squeeze(-1)
