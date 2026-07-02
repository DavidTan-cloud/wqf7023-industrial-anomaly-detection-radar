import numpy as np

def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size):
        windows.append(
            data[i:i + window_size]
        )

    return np.array(windows)
