import sys
from pathlib import Path as _Path
sys.path.insert(0,str(_Path(__file__).resolve().parents[1]))
import torch, random, numpy as np
from mini_llm.tokenizer import SimpleTokenizer
from mini_llm.model import MiniLLM
from mini_llm.loss import causal_lm_loss
from mini_jev.decision_model import MiniJev
from mini_jev.inference import predict

def main():
 random.seed(42);np.random.seed(42);torch.manual_seed(42); texts=['state transfer failed','Which intent best matches this state?','failed_transfer','card_arrival','low','high','yes']
 tok=SimpleTokenizer().fit(texts); llm=MiniLLM(tok.vocab_size); x=torch.tensor([tok.encode('state: transfer failed answer: failed_transfer',64)]); loss=causal_lm_loss(llm(x),x); loss.backward(); assert torch.isfinite(loss)
 jev=MiniJev(tok.vocab_size); qs=[{'type':'choice','instruction':'Which intent best matches this state?','options':['failed_transfer','card_arrival']},{'type':'score','instruction':'How urgent is this?','levels':['low','medium','high']},{'type':'noul','instruction':'Should this be escalated?'}]; out=predict(jev,tok,'transfer failed',qs)
 assert abs(float(out[0]['probabilities'].sum().detach())-1)<1e-5; assert abs(float(out[1]['probabilities'].sum().detach())-1)<1e-5; assert 0<=out[2]['probability_yes']<=1; print('smoke test passed')
if __name__=='__main__':main()
