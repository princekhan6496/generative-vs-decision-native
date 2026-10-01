import numpy as np
def multiclass_brier(probs,targets):
 p=np.asarray(probs); t=np.zeros_like(p); t[np.arange(len(targets)),targets]=1; return float(np.mean(np.sum((p-t)**2,axis=1)))
def binary_brier(probs,targets): p=np.asarray(probs); return float(np.mean((p-np.asarray(targets))**2))
