import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from mini_llm.tokenizer import SimpleTokenizer
from mini_llm.model import MiniLLM
from mini_llm.loss import causal_lm_loss
from mini_jev.decision_model import MiniJev
from mini_jev.inference import prepare_question

def main():
 texts=['transfer failed','card arrived','failed_transfer','card_arrival','Which intent?','urgent','not urgent']
 tok=SimpleTokenizer().fit(texts); device=torch.device('cpu')
 llm=MiniLLM(tok.vocab_size,d_model=32,n_heads=4,n_layers=1,d_ff=64,max_seq_len=64).to(device); opt=torch.optim.AdamW(llm.parameters(),lr=1e-3)
 x=torch.tensor([tok.encode('state transfer failed answer failed_transfer',32)],device=device)
 llm.train()
 for _ in range(3):
  loss=causal_lm_loss(llm(x),x); opt.zero_grad(); loss.backward(); opt.step()
 jev=MiniJev(tok.vocab_size,d_model=32,n_heads=4,n_layers=1,d_ff=64,max_seq_len=64).to(device); opt=torch.optim.AdamW(jev.parameters(),lr=1e-3)
 q={'type':'choice','instruction':'Which intent?','options':['failed_transfer','card_arrival']}; ids=torch.tensor([tok.encode('transfer failed',32)],device=device); qd=[[prepare_question(q,tok,32,device)]]; target=torch.tensor([0],device=device)
 jev.train()
 for _ in range(3):
  pred=jev(ids,qd)[0][0]; loss=torch.nn.functional.cross_entropy(pred['logits'].unsqueeze(0),target); opt.zero_grad(); loss.backward(); opt.step()
 print('tiny end-to-end training passed; final LLM loss:',float(loss.detach())); print('Mini-LLM params:',sum(p.numel() for p in llm.parameters())); print('Mini-Jev params:',sum(p.numel() for p in jev.parameters()))
if __name__=='__main__': main()
