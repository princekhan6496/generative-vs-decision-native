import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse,json,time
from pathlib import Path
import torch
from mini_llm.tokenizer import SimpleTokenizer
from mini_llm.model import MiniLLM
from mini_llm.loss import causal_lm_loss

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--text',required=True); ap.add_argument('--epochs',type=int,default=1); ap.add_argument('--max-samples',type=int,default=1000); ap.add_argument('--out',default='results/tinystories.json'); a=ap.parse_args()
    lines=[x.strip() for x in open(a.text,encoding='utf8') if x.strip()][:a.max_samples]; tok=SimpleTokenizer().fit(lines); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MiniLLM(tok.vocab_size,d_model=96,n_heads=4,n_layers=2,d_ff=192,max_seq_len=128).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=5e-4); start=time.perf_counter(); losses=[]
    x=torch.tensor([tok.encode(s,128) for s in lines],device=dev)
    for _ in range(a.epochs):
        for i in range(0,len(x),16):
            loss=causal_lm_loss(model(x[i:i+16]),x[i:i+16]); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); losses.append(float(loss.detach()))
    result={'task':'language_generation','dataset':'TinyStories','examples':len(lines),'training_seconds':time.perf_counter()-start,'final_loss':losses[-1] if losses else None,'status':'measured'}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(result,open(a.out,'w'),indent=2); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
