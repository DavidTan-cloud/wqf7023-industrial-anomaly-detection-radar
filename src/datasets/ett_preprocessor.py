import os
import sys
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from src.preprocessing.normalization import (
    DataNormalizer
)

from src.preprocessing.windowing import (
    create_windows
)

from src.preprocessing.label_windowing import (
    create_window_labels
)

np.random.seed(42)

files = [
    "ETTh1.csv",
    "ETTh2.csv",
    "ETTm1.csv",
    "ETTm2.csv"
]

WINDOW_SIZE = 100

os.makedirs(
    "src/datasets/processed/ETT",
    exist_ok=True
)

for filename in files:

    print(f"\nProcessing {filename}")

    df = pd.read_csv(
        f"src/datasets/raw/ETT/{filename}"
    )

    if "date" in df.columns:
        df = df.drop(
            columns=["date"]
        )

    data = df.values.astype(
        np.float32
    )

    n = len(data)

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    train = data[:train_end]

    val = data[
        train_end:val_end
    ]

    test = data[val_end:]

    scaler = DataNormalizer()

    train = scaler.fit_transform(
        train
    )

    val = scaler.transform(
        val
    )

    test = scaler.transform(
        test
    )

    # Create synthetic anomalies

    y_test = np.zeros(
        len(test),
        dtype=np.int8
    )

    anomaly_ratio = 0.10

    n_anomalies = int(
        len(test)
        * anomaly_ratio
    )

    anomaly_idx = np.random.choice(
        len(test),
        n_anomalies,
        replace=False
    )

    y_test[anomaly_idx] = 1

    test_anomaly = test.copy()

    test_anomaly[
        anomaly_idx
    ] += np.random.normal(
        loc=5.0,
        scale=1.0,
        size=test_anomaly[
            anomaly_idx
        ].shape
    )

    X_train = create_windows(
        train,
        WINDOW_SIZE
    )

    X_val = create_windows(
        val,
        WINDOW_SIZE
    )

    X_test = create_windows(
        test_anomaly,
        WINDOW_SIZE
    )

    y_test_window = create_window_labels(
        y_test,
        WINDOW_SIZE
    )

    base_name = filename.replace(
        ".csv",
        ""
    )

    np.save(
        f"src/datasets/processed/ETT/{base_name}_X_train.npy",
        X_train
    )

    np.save(
        f"src/datasets/processed/ETT/{base_name}_X_val.npy",
        X_val
    )

    np.save(
        f"src/datasets/processed/ETT/{base_name}_X_test.npy",
        X_test
    )

    np.save(
        f"src/datasets/processed/ETT/{base_name}_y_test.npy",
        y_test_window
    )

    joblib.dump(
        scaler,
        f"src/datasets/processed/ETT/{base_name}_scaler.pkl"
    )

    print(
        f"Completed {base_name}"
    )

    print(
        "Train:",
        X_train.shape
    )

    print(
        "Val:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    print(
        "Anomaly Windows:",
        np.sum(
            y_test_window == 1
        )
    )
