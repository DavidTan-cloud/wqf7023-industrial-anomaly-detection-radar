import os
import sys
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

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
from src.preprocessing.normalization import (
    DataNormalizer
)

# Create loader
loader = MIMIILoader(
    root_dir="src/datasets/raw/MIMII"
)

# Create processor
processor = MIMIIPreprocessor()

# Load files
normal_files, anomaly_files = loader.load_machine(
    dataset_name="0_dB_fan",
    machine_id="00"
)

print(
    f"Normal files: {len(normal_files)}"
)

print(
    f"Anomaly files: {len(anomaly_files)}"
)

train_normal, temp_normal = train_test_split(
    normal_files,
    test_size=0.30,
    random_state=42
)

val_normal, test_normal = train_test_split(
    temp_normal,
    test_size=0.50,
    random_state=42
)

X_train = []

for file in train_normal:

    feature = processor.extract_features(
        file
    )

    X_train.append(
        feature
    )

X_train = np.array(
    X_train
)

test_files = (
    test_normal
    + anomaly_files
)

X_test = []

for file in test_files:

    feature = processor.extract_features(
        file
    )

    X_test.append(
        feature
    )

X_test = np.array(
    X_test
)

y_train = np.zeros(
    len(X_train)
)

y_test = []

for file in test_files:

    if "abnormal" in file:
        y_test.append(1)
    else:
        y_test.append(0)

y_test = np.array(
    y_test
)

scaler = DataNormalizer()

X_train_2d = X_train.reshape(
    -1,
    X_train.shape[-1]
)

X_train_2d = scaler.fit_transform(
    X_train_2d
)

X_train = X_train_2d.reshape(
    X_train.shape
)

X_test_2d = X_test.reshape(
    -1,
    X_test.shape[-1]
)

X_test_2d = scaler.transform(
    X_test_2d
)

X_test = X_test_2d.reshape(
    X_test.shape
)

X_val = []

for file in val_normal:

    feature = processor.extract_features(
        file
    )

    X_val.append(
        feature
    )

X_val = np.array(
    X_val
)

y_val = np.zeros(
    len(X_val)
)

X_val_2d = X_val.reshape(
    -1,
    X_val.shape[-1]
)

X_val_2d = scaler.transform(
    X_val_2d
)

X_val = X_val_2d.reshape(
    X_val.shape
)

os.makedirs(
    "src/datasets/processed/MIMII",
    exist_ok=True
)

np.save(
    "src/datasets/processed/MIMII/fan_id00_X_train.npy",
    X_train
)

np.save(
    "src/datasets/processed/MIMII/fan_id00_y_train.npy",
    y_train
)

np.save(
    "src/datasets/processed/MIMII/fan_id00_X_test.npy",
    X_test
)

np.save(
    "src/datasets/processed/MIMII/fan_id00_y_test.npy",
    y_test
)

np.save(
    "src/datasets/processed/MIMII/fan_id00_X_val.npy",
    X_val
)

np.save(
    "src/datasets/processed/MIMII/fan_id00_y_val.npy",
    y_val
)

joblib.dump(
    scaler,
    "src/datasets/processed/MIMII/fan_id00_scaler.pkl"
)

print("Train Normal:", len(train_normal))
print("Validation Normal:", len(val_normal))
print("Test Normal:", len(test_normal))
print("Anomaly:", len(anomaly_files))

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)

print(
    "Test Normals:",
    np.sum(y_test == 0)
)

print(
    "Test Anomalies:",
    np.sum(y_test == 1)
)

print(
    f"Normal files: {len(normal_files)}"
)

print(
    f"Anomaly files: {len(anomaly_files)}"
)