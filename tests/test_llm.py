import torch
from mini_llm.model import MiniLLM
def test_shapes_and_causal():
 m=MiniLLM(20,d_model=32,n_heads=4,n_layers=1,d_ff=64); x=torch.randint(0,20,(2,8)); y=m(x); assert y.shape==(2,8,20); assert torch.isfinite(y).all()
