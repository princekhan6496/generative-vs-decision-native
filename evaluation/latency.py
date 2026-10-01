import time, torch

def measure(fn, warmup=3, runs=20):
    for _ in range(warmup): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(runs): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.perf_counter()-start)/runs*1000

def throughput(fn, batch_size, runs=20):
    for _ in range(3): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(runs): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    return batch_size*runs/elapsed
