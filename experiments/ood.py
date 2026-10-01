import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse,json,random
from pathlib import Path
import numpy as np, torch
from mini_llm.tokenizer import SimpleTokenizer
from mini_llm.model import MiniLLM
from mini_jev.decision_model import MiniJev
from datasets.preprocessing import banking77_to_records,deterministic_split,shifted_records,make_decision_example
from experiments.benchmark import train_llm,train_jev,llm_arrays,choice_arrays,seed,make_tokenizer
from evaluation.benchmark import evaluate_classification

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--banking-train',required=True); ap.add_argument('--banking-test',required=True); ap.add_argument('--epochs',type=int,default=1); ap.add_argument('--limit',type=int); ap.add_argument('--out',default='results/ood.json'); a=ap.parse_args()
    seed(42); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw=banking77_to_records(a.banking_train); tr,_,_=deterministic_split(raw,42); te=banking77_to_records(a.banking_test)
    if a.limit: tr=tr[:a.limit]; te=te[:a.limit]
    labels=sorted({r['label'] for r in tr}); te=[r for r in te if r['label'] in labels]; shifted=shifted_records(te)
    labels=sorted({r['label'] for r in tr}); tokenizer_examples=[make_decision_example(r,labels) for r in tr]; tok=make_tokenizer(tokenizer_examples); cfg={'d_model':96,'n_heads':4,'n_layers':2,'d_ff':192,'dropout':.1,'max_seq_len':128}
    train=[make_decision_example(r,labels) for r in tr]; original=[make_decision_example(r,labels) for r in te]; ood=[make_decision_example(r,labels) for r in shifted]
    llm=MiniLLM(tok.vocab_size,**cfg).to(dev); jev=MiniJev(tok.vocab_size,**cfg).to(dev); train_llm(llm,tok,train,a.epochs,dev,5e-4,16); train_jev(jev,tok,train,a.epochs,dev,5e-4,16)
    y,pl,_=llm_arrays(llm,tok,original,dev,0); yo,plo,_=llm_arrays(llm,tok,ood,dev,0); yj,pj,_=choice_arrays(jev,tok,original,dev,0); yjo,pjo,_=choice_arrays(jev,tok,ood,dev,0)
    result={'protocol':'train only on original distribution; no OOD tuning','original':{'mini_llm':evaluate_classification(y,pl),'mini_jev':evaluate_classification(yj,pj)},'shifted':{'mini_llm':evaluate_classification(yo,plo),'mini_jev':evaluate_classification(yjo,pjo)},'degradation':{'accuracy':{'mini_llm':evaluate_classification(y,pl)['accuracy']-evaluate_classification(yo,plo)['accuracy'],'mini_jev':evaluate_classification(yj,pj)['accuracy']-evaluate_classification(yjo,pjo)['accuracy']},'ece':{'mini_llm':evaluate_classification(yo,plo)['ece']-evaluate_classification(y,pl)['ece'],'mini_jev':evaluate_classification(yjo,pjo)['ece']-evaluate_classification(yj,pj)['ece']}},'status':'measured'}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(result,open(a.out,'w'),indent=2); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
