import numpy as np

def mutate(theta, sigma=0.1, bounds=None):
    g = np.array(theta, dtype=float) + np.random.randn(*np.asarray(theta).shape) * sigma
    if bounds is not None:
        low, high = bounds
        g = np.clip(g, low, high)
    return g

def evo_search(init_thetas, evaluate_fn, iters=50, pop=64, keep=16, sigma=0.2, bounds=None):
    """(μ+λ) evolution: sample → mutate → select by score (higher is better).
    Returns (pool, scores, best_theta, best_score).
    """
    pool = list(init_thetas)
    scores = [evaluate_fn(t) for t in pool]
    for _ in range(iters):
        children = [mutate(pool[np.random.randint(len(pool))], sigma, bounds) for __ in range(pop)]
        child_scores = [evaluate_fn(c) for c in children]
        pool = pool + children
        scores = scores + child_scores
        order = np.argsort(scores)[::-1][:keep]
        pool = [pool[i] for i in order]
        scores = [scores[i] for i in order]
    best_idx = int(np.argmax(scores))
    return pool, scores, pool[best_idx], scores[best_idx]
