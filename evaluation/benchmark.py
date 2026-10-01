import numpy as np
import json,torch
from .accuracy import accuracy
from .f1 import macro_f1,precision_recall
from .brier import multiclass_brier
from .ece import ece
def parameter_count(m): return sum(p.numel() for p in m.parameters())
def evaluate_classification(y,probs):
 p=probs.argmax(1).tolist(); return {'accuracy':accuracy(y,p),'macro_f1':macro_f1(y,p),'precision_macro':precision_recall(y,p)[0],'recall_macro':precision_recall(y,p)[1],'brier':multiclass_brier(probs,y),'ece':ece(probs,y),'nll':float(torch.nn.functional.nll_loss(torch.tensor(probs).clamp_min(1e-8).log(),torch.tensor(y)))}
def score_metrics(probs, targets):
 p=np.asarray(probs); t=np.asarray(targets); values=np.arange(1,p.shape[1]+1,dtype=float); expected=(p*values).sum(1); target_values=t.astype(float)+1; return {'mae':float(np.mean(np.abs(expected-target_values))),'expected_score_mean':float(expected.mean()),'target_score_mean':float(target_values.mean()),'expected_scores':expected.tolist()}

def binary_metrics(prob_yes, targets):
 p=np.asarray(prob_yes,dtype=float); t=np.asarray(targets,dtype=int); pred=(p>=0.5).astype(int); return {'accuracy':float(np.mean(pred==t)),'brier':float(np.mean((p-t)**2)),'nll':float(-np.mean(t*np.log(np.clip(p,1e-8,1))+(1-t)*np.log(np.clip(1-p,1e-8,1))))}

def save_results(path,obj): json.dump(obj,open(path,'w'),indent=2)
