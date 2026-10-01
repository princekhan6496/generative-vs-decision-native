import numpy as np
def macro_f1(y,p):
 labs=sorted(set(y)|set(p)); vals=[]
 for c in labs:
  tp=sum(a==c and b==c for a,b in zip(y,p)); fp=sum(a!=c and b==c for a,b in zip(y,p)); fn=sum(a==c and b!=c for a,b in zip(y,p)); pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0; vals.append(2*pr*re/(pr+re) if pr+re else 0)
 return float(np.mean(vals))
def precision_recall(y,p):
 labs=sorted(set(y)|set(p)); ps=[];rs=[]
 for c in labs:
  tp=sum(a==c and b==c for a,b in zip(y,p)); fp=sum(a!=c and b==c for a,b in zip(y,p)); fn=sum(a==c and b!=c for a,b in zip(y,p)); ps.append(tp/(tp+fp) if tp+fp else 0); rs.append(tp/(tp+fn) if tp+fn else 0)
 return float(np.mean(ps)),float(np.mean(rs))
