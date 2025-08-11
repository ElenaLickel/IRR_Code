import numpy as np

def brier_score(p, y_true):
    y_onehot = np.zeros_like(p)
    y_onehot[np.arange(len(y_true)), y_true] = 1
    return np.mean(np.sum((y_onehot - p)**2, axis=1))

def spherical_score(p, y_true):
    norms = np.linalg.norm(p, axis=1)
    return np.mean([p[i, y_true[i]] / norms[i] for i in range(len(y_true))])

def log_score(p, y_true):
    return -np.mean([np.log(p[i, y_true[i]] + 1e-15) for i in range(len(y_true))])
