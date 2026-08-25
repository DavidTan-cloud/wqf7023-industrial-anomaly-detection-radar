import os
import glob
import librosa
import numpy as np

class MIMIIPreprocessor:

    def __init__(
        self,
        n_mfcc=20,
        sample_rate=16000
    ):
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate

    def extract_features(
        self,
        filepath
    ):
        audio, sr = librosa.load(
            filepath,
            sr=self.sample_rate,
            mono=True
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc
        )

        return mfcc.T

    def process_directory(
        self,
        normal_dir,
        anomaly_dir
    ):
        X = []
        y = []

        normal_files = sorted(
            glob.glob(
                os.path.join(
                    normal_dir,
                    "*.wav"
                )
            )
        )

        anomaly_files = sorted(
            glob.glob(
                os.path.join(
                    anomaly_dir,
                    "*.wav"
                )
            )
        )

        for file in normal_files:

            features = self.extract_features(
                file
            )

            X.append(features)
            y.append(0)

        for file in anomaly_files:

            features = self.extract_features(
                file
            )

            X.append(features)
            y.append(1)

        return X, np.array(y)
