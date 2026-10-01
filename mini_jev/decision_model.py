import torch, torch.nn as nn
from .encoder import StateEncoder
from .question_encoder import QuestionEncoder
from .choice_head import ChoiceHead
from .score_head import ScoreHead
from .noul_head import NoulHead
from .calibration import confidence
class MiniJev(nn.Module):
    def __init__(self,vocab_size,d_model=96,n_heads=4,n_layers=2,d_ff=192,dropout=.1,max_seq_len=128):
        super().__init__(); self.encoder=StateEncoder(vocab_size,d_model,n_heads,n_layers,d_ff,dropout,max_seq_len); self.qenc=QuestionEncoder(vocab_size,d_model); self.choice=ChoiceHead(d_model); self.score=ScoreHead(d_model); self.noul=NoulHead(d_model)
    def forward(self,state_ids,questions):
        shared=self.encoder(state_ids); outputs=[]
        for i,qs in enumerate(questions):
            sample=[]
            for q in qs:
                qids=q['ids'].unsqueeze(0) if q['ids'].dim()==1 else q['ids']; qr=self.qenc(qids,q['type']).squeeze(0); h=shared[i]; z=h+qr
                if q['type']=='choice':
                    opts=[self.qenc(o.unsqueeze(0) if o.dim()==1 else o,'choice').squeeze(0) for o in q['option_ids']]; logits=self.choice(z,opts); p=logits.softmax(-1); sample.append({'type':'choice','logits':logits,'probabilities':p,'selected':int(p.argmax()),'confidence':confidence(p)})
                elif q['type']=='score':
                    logits=self.score(h,qr,len(q['levels'])); p=logits.softmax(-1); vals=torch.tensor(q['values'],device=p.device,dtype=p.dtype); sample.append({'type':'score','logits':logits,'probabilities':p,'score':float((p*vals).sum().detach()),'confidence':confidence(p)})
                else:
                    logit=self.noul(h,qr); sample.append({'type':'noul','logit':logit,'probability_yes':float(logit.sigmoid().detach())})
            outputs.append(sample)
        return outputs
