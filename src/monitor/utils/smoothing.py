import numpy as np

def moving_avg(vals, k):
    if not vals:
        return 0.0
    if len(vals) < k:
        return float(np.mean(vals))
    return float(np.mean(list(vals)[-k:]))
