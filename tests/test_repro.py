import torch
from mini_llm.model import MiniLLM
def test_fixed_seed():
 torch.manual_seed(7); a=MiniLLM(10,d_model=16,n_heads=4,n_layers=1,d_ff=32); a.eval(); x=torch.randint(0,10,(1,4)); y=a(x); torch.manual_seed(7); b=MiniLLM(10,d_model=16,n_heads=4,n_layers=1,d_ff=32); b.eval(); z=b(x); assert torch.allclose(y,z)
