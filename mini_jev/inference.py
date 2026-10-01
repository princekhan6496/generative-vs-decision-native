import torch
def prepare_question(q,tokenizer,max_len=64,device=None):
    d={'type':q['type'],'ids':torch.tensor(tokenizer.encode(q['instruction'],max_len),dtype=torch.long,device=device)}
    if q['type']=='choice': d['option_ids']=[torch.tensor(tokenizer.encode(x,max_len),dtype=torch.long,device=device) for x in q['options']]
    if q['type']=='score': d['levels']=q['levels']; d['values']=list(range(1,len(q['levels'])+1))
    return d
def predict(model,tokenizer,state,questions,max_len=128):
    dev=next(model.parameters()).device; ids=torch.tensor([tokenizer.encode(state,max_len)],dtype=torch.long,device=dev); qs=[[prepare_question(q,tokenizer,64,dev) for q in questions]]; return model(ids,qs)[0]
