# src/datasets/test_mfcc.py

import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.datasets.mimii_preprocessor import (
    MIMIIPreprocessor
)

processor = MIMIIPreprocessor()

features = processor.extract_features(
    "src/datasets/raw/MIMII/0_dB_fan/id_00/normal/normal_id_00_00000000.wav"
)

print(features.shape)