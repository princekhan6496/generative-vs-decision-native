import numpy as np

def reliability_data(probs, targets, n_bins=10):
    probs = np.asarray(probs); targets = np.asarray(targets)
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); correct = (pred == targets).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); rows=[]
    for i in range(n_bins):
        mask = (conf >= edges[i]) & ((conf < edges[i+1]) if i < n_bins-1 else (conf <= edges[i+1]))
        if mask.any(): rows.append({'bin': i, 'count': int(mask.sum()), 'confidence': float(conf[mask].mean()), 'accuracy': float(correct[mask].mean())})
    return rows

def confidence_histogram(probs, bins=10):
    counts, edges = np.histogram(np.asarray(probs).max(axis=1), bins=bins, range=(0,1))
    return {'counts': counts.tolist(), 'edges': edges.tolist()}
