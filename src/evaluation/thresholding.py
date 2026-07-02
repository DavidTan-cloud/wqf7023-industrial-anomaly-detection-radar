import numpy as np

def percentile_threshold(
    scores,
    percentile=95
):

    return np.percentile(
        scores,
        percentile
    )
