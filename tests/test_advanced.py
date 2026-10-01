import torch
from mini_llm.rope import apply_rope
from mini_llm.tokenizer import SimpleTokenizer
from mini_jev.decision_model import MiniJev
from mini_jev.inference import prepare_question

def test_rope_preserves_shape_and_finite():
    q=torch.randn(2,4,7,8); k=torch.randn_like(q); a,b=apply_rope(q,k); assert a.shape==q.shape and b.shape==k.shape and torch.isfinite(a).all()

def test_multi_question_forward_shared_state():
    tok=SimpleTokenizer().fit(['hello','choice one','low','high','is this true'])
    m=MiniJev(tok.vocab_size,d_model=32,n_heads=4,n_layers=1,d_ff=64,max_seq_len=32)
    state=torch.tensor([tok.encode('hello',32)])
    qs=[[prepare_question({'type':'choice','instruction':'choose','options':['choice one','hello']},tok,16), prepare_question({'type':'score','instruction':'rate','levels':['low','high']},tok,16), prepare_question({'type':'noul','instruction':'is this true'},tok,16)]]
    out=m(state,qs)[0]; assert len(out)==3; assert torch.allclose(out[0]['probabilities'].sum(),torch.tensor(1.)); assert torch.allclose(out[1]['probabilities'].sum(),torch.tensor(1.)); assert 0<=out[2]['probability_yes']<=1
