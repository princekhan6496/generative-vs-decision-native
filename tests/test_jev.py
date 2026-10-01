import torch
from mini_llm.tokenizer import SimpleTokenizer
from mini_jev.decision_model import MiniJev
from mini_jev.inference import predict
def test_typed_outputs_and_parallel_questions():
 t=SimpleTokenizer().fit(['hello','choice','score','noul','a','b','low','high','yes']); m=MiniJev(t.vocab_size,d_model=32,n_heads=4,n_layers=1,d_ff=64); qs=[{'type':'choice','instruction':'choice','options':['a','b']},{'type':'score','instruction':'score','levels':['low','high']},{'type':'noul','instruction':'noul'}]; out=predict(m,t,'hello',qs); assert len(out)==3; assert torch.isclose(out[0]['probabilities'].sum(),torch.tensor(1.),atol=1e-5); assert torch.isclose(out[1]['probabilities'].sum(),torch.tensor(1.),atol=1e-5); assert 0<=out[2]['probability_yes']<=1
