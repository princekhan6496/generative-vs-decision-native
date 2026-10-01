import torch.nn as nn
class EncoderSelfAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout=.1):
        super().__init__(); self.attn=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True)
    def forward(self,x,key_padding_mask=None): return self.attn(x,x,x,key_padding_mask=key_padding_mask,need_weights=False)[0]
