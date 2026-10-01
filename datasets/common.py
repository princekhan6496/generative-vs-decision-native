import torch
from torch.utils.data import Dataset
class LabeledTextDataset(Dataset):
 def __init__(self,records,tokenizer,max_len): self.x=[torch.tensor(tokenizer.encode(r['state'],max_len),dtype=torch.long) for r in records]; self.y=[r['label'] for r in records]
 def __len__(self): return len(self.x)
 def __getitem__(self,i): return self.x[i],self.y[i]
def collate(items): return torch.stack([x for x,_ in items]),[y for _,y in items]
