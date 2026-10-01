import torch, torch.nn as nn
from .embeddings import TokenEmbedding
from .transformer import DecoderTransformer
class MiniLLM(nn.Module):
    def __init__(self,vocab_size,d_model=96,n_heads=4,n_layers=2,d_ff=192,dropout=.1,max_seq_len=128):
        super().__init__(); self.max_seq_len=max_seq_len; self.emb=TokenEmbedding(vocab_size,d_model); self.tr=DecoderTransformer(d_model,n_heads,n_layers,d_ff,dropout); self.lm_head=nn.Linear(d_model,vocab_size,bias=False); self.lm_head.weight=self.emb.embedding.weight
    def forward(self,input_ids): return self.lm_head(self.tr(self.emb(input_ids)))
    @torch.no_grad()
    def generate(self,input_ids,max_new_tokens=40,temperature=1.,top_k=None,top_p=None,eos_id=2,pad_id=0,bos_id=1):
        self.eval(); x=input_ids
        for _ in range(max_new_tokens):
            logits=self(x[:,-self.max_seq_len:])[:,-1]/max(temperature,1e-6)
            # PAD and BOS are input/control symbols, not valid generated answer
            # tokens.  EOS is left available and is the only early-stop token.
            if 0 <= pad_id < logits.size(-1): logits[:,pad_id]=-torch.inf
            if 0 <= bos_id < logits.size(-1): logits[:,bos_id]=-torch.inf
            if top_k:
                v,ix=torch.topk(logits,min(top_k,logits.size(-1))); z=torch.full_like(logits,-torch.inf); z.scatter_(1,ix,v); logits=z
            if top_p:
                s,ix=torch.sort(logits,descending=True); p=torch.softmax(s,-1); cut=(torch.cumsum(p,-1)-p)>top_p; s[cut]=-torch.inf; logits=torch.full_like(logits,-torch.inf).scatter(1,ix,s)
            nxt=torch.argmax(logits,-1,keepdim=True); x=torch.cat([x,nxt],1)
            if (nxt==eos_id).all(): break
        return x
