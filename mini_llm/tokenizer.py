import re, json
from collections import Counter

SPECIAL=["<pad>","<bos>","<eos>","<unk>"]
class SimpleTokenizer:
    def __init__(self,vocab=None): self.itos=vocab or SPECIAL[:]; self.stoi={t:i for i,t in enumerate(self.itos)}
    def fit(self,texts,min_freq=1):
        c=Counter();
        for s in texts: c.update(re.findall(r"[A-Za-z0-9_]+|[^\w\s]", s.lower()))
        for t,n in c.most_common():
            if n>=min_freq and t not in self.stoi: self.stoi[t]=len(self.itos); self.itos.append(t)
        return self
    def tokenize(self,s): return re.findall(r"[A-Za-z0-9_]+|[^\w\s]", s.lower())
    def encode(self,s,max_length=None,add_special=True):
        ids=[self.stoi.get(t,self.stoi["<unk>"]) for t in self.tokenize(s)]
        if add_special: ids=[self.stoi["<bos>"]]+ids+[self.stoi["<eos>"]]
        if max_length: ids=ids[:max_length]; ids += [self.stoi["<pad>"]]*max(0,max_length-len(ids))
        return ids
    def decode(self,ids): return " ".join(self.itos[i] for i in ids if self.itos[i] not in SPECIAL)
    def save(self,path): json.dump(self.itos,open(path,'w',encoding='utf8'))
    @classmethod
    def load(cls,path): return cls(json.load(open(path,encoding='utf8')))
    @property
    def vocab_size(self): return len(self.itos)
