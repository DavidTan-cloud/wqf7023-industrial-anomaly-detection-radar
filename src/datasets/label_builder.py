import numpy as np
import ast

def create_labels(
    anomaly_sequence,
    sequence_length
):

    labels = np.zeros(
        sequence_length,
        dtype=int
    )

    intervals = ast.literal_eval(
        anomaly_sequence
    )

    for start, end in intervals:
        labels[start:end] = 1
    return labels
