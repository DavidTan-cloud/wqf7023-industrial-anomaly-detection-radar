"""
SMAP Dataset Loader
Author: David Tan

Loads NASA SMAP telemetry dataset and prepares
train/test splits for anomaly detection.
"""
import os
import numpy as np
import pandas as pd

class SMAPLoader:

    
def __init__(
        self,
        train_dir,
        test_dir,
        labels_file
    ):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.labels_file = labels_file

    def get_channels(self):
        return sorted(os.listdir(self.train_dir))

    def load_channel(self, channel):
        train = np.load(
            os.path.join(
                self.train_dir,
                channel
            )
        )

        test = np.load(
            os.path.join(
                self.test_dir,
                channel
            )
        )
        return train, test

    def load_labels(self):
        return pd.read_excel(
            self.labels_file
        )
