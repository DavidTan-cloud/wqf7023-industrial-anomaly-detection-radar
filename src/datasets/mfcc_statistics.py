import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.datasets.mimii_loader import MIMIILoader
from src.datasets.mimii_preprocessor import (
    MIMIIPreprocessor
)

loader = MIMIILoader(
    root_dir="src/datasets/raw/MIMII"
)

processor = MIMIIPreprocessor()

normal_files, anomaly_files = loader.load_machine(
    dataset_name="0_dB_fan",
    machine_id="00"
)

lengths = []

for file in normal_files[:100]:

    feature = processor.extract_features(
        file
    )

    lengths.append(
        feature.shape[0]
    )

print(
    "Min:",
    min(lengths)
)

print(
    "Mean:",
    np.mean(lengths)
)

print(
    "Max:",
    max(lengths)
)
