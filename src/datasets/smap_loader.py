"""
SMAP Dataset Loader
Author: David Tan

Loads NASA SMAP telemetry dataset and prepares
train/test splits for anomaly detection.
"""
import pandas as pd
import numpy as np

class SMAPLoader:

    def __init__(self,
                 train_path,
                 test_path,
                 label_path=None):

        self.train_path = train_path
        self.test_path = test_path
        self.label_path = label_path

    def load(self):

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        labels = None

        if self.label_path is not None:
            labels = pd.read_csv(self.label_path)

        return train_df, test_df, labels

    def summary(self, df):

        print("Shape:", df.shape)
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nMissing Values:")
        print(df.isnull().sum())
