import torch

def confidence(p):
    if p.numel() <= 1:
        return torch.tensor(1., device=p.device)
    p = p.clamp_min(1e-8)
    e = -(p * p.log()).sum()
    return 1 - e / torch.log(torch.tensor(float(p.numel()), device=p.device))

def temperature_scale(logits, temperature):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    return logits / temperature

def fit_temperature(logits, targets, max_iter=50):
    logits = logits.detach()
    targets = targets.detach()
    log_t = torch.zeros((), device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=.1, max_iter=max_iter, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp().clamp_min(1e-3), targets)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp().detach().clamp_min(1e-3))
