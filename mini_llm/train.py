import time, torch
from torch.utils.data import DataLoader,Dataset
from .loss import causal_lm_loss
class LMDataset(Dataset):
    def __init__(self,ids): self.ids=torch.tensor(ids,dtype=torch.long)
    def __len__(self): return len(self.ids)
    def __getitem__(self,i): return self.ids[i]
def train(model,loader,epochs,lr,weight_decay,grad_clip,device,checkpoint=None):
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay); start=time.time()
    for _ in range(epochs):
        model.train()
        for x in loader:
            x=x.to(device); loss=causal_lm_loss(model(x),x); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip); opt.step()
    if checkpoint: torch.save({'model':model.state_dict()},checkpoint)
    return {'training_seconds':time.time()-start}
