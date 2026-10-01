def accuracy(y,p): return sum(a==b for a,b in zip(y,p))/max(1,len(y))
