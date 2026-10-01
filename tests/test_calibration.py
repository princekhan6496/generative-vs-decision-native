import torch
from evaluation.ece import ece
from evaluation.brier import multiclass_brier
def test_metrics():
 p=torch.tensor([[.9,.1],[.2,.8]]).numpy(); y=[0,1]; assert multiclass_brier(p,y)>=0; assert 0<=ece(p,y)<=1
