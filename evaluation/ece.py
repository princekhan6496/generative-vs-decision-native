import numpy as np
def ece(probs,targets,bins=10):
 p=np.asarray(probs); t=np.asarray(targets); conf=p.max(1); pred=p.argmax(1); acc=(pred==t).astype(float); out=0
 for lo,hi in zip(np.linspace(0,1,bins,endpoint=False),np.linspace(0,1,bins+1)):
  m=(conf>=lo)&(conf<hi if hi<1 else conf<=hi)
  if m.any(): out += m.mean()*abs(acc[m].mean()-conf[m].mean())
 return float(out)
def reliability(probs,targets,bins=10):
 p=np.asarray(probs);t=np.asarray(targets); c=p.max(1);a=(p.argmax(1)==t); rows=[]
 for lo,hi in zip(np.linspace(0,1,bins,endpoint=False),np.linspace(0,1,bins+1)):
  m=(c>=lo)&(c<hi if hi<1 else c<=hi)
  if m.any(): rows.append((float(c[m].mean()),float(a[m].mean()),int(m.sum())))
 return rows
