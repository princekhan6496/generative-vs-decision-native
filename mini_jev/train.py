import time
import torch
import torch.nn.functional as F

class DecisionTrainer:
    def __init__(self, model, lr=.0005, weight_decay=.01, grad_clip=1.):
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.clip = grad_clip

    def step(self, batch, device):
        self.model.train()
        state = batch['state'].to(device)
        questions = batch['questions']
        outputs = self.model(state, questions)
        total = torch.zeros((), device=device)
        count = 0
        for sample_out, sample_targets in zip(outputs, batch['targets']):
            for pred, target in zip(sample_out, sample_targets):
                kind = target['type']
                if kind in ('choice', 'score'):
                    t = torch.tensor([target['target']], device=device)
                    total = total + F.cross_entropy(pred['logits'].unsqueeze(0), t)
                elif kind == 'noul':
                    t = torch.tensor([float(target['target'])], device=device)
                    total = total + F.binary_cross_entropy_with_logits(pred['logit'].view(1), t)
                count += 1
        if count == 0:
            raise ValueError('Batch contains no decision targets')
        loss = total / count
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.opt.step()
        return float(loss.detach())

    def fit(self, batches, epochs, device):
        self.model.to(device)
        start = time.time()
        history = []
        for _ in range(epochs):
            for batch in batches:
                history.append(self.step(batch, device))
        return {'training_seconds': time.time() - start, 'loss_history': history}
