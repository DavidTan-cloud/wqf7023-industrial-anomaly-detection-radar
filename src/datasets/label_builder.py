import numpy as np
import ast

def build_labels(
    anomaly_sequence,
    length
):

    labels = np.zeros(
        length,
        dtype=int
    )

    intervals = ast.literal_eval(
        anomaly_sequence
    )

    for start, end in intervals:
        labels[start:end] = 1
    return labels
