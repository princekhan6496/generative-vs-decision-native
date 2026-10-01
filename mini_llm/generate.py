import torch,json
def generate_decision(model,tokenizer,prompt,max_new_tokens=40):
    ids=torch.tensor([tokenizer.encode(prompt,add_special=True)],dtype=torch.long,device=next(model.parameters()).device); out=model.generate(ids,max_new_tokens=max_new_tokens,temperature=.7,top_p=.9); text=tokenizer.decode(out[0].tolist()); return text
