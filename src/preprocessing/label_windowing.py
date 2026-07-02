import numpy as np

def create_window_labels(
    labels,
    window_size
):

    y = []
    for i in range(
        len(labels) - window_size
    ):

        y.append(
            int(
                labels[
                    i:i+window_size
                ].max()
            )
        )

    return np.array(y)
