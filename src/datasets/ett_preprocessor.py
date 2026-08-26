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

    test_anomaly = test.copy()

    num_blocks = 5
    block_size = 10

    for _ in range(num_blocks):

        start = np.random.randint(
            0,
            len(test) - block_size
        )

        y_test[
            start:start+block_size
        ] = 1

        test_anomaly[
            start:start+block_size
        ] += np.random.normal(
            loc=5.0,
            scale=1.0,
            size=test_anomaly[
                start:start+block_size
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

    print(
        "Point Anomalies:",
        np.sum(y_test == 1)
    )

    print(
        "Point Normals:",
        np.sum(y_test == 0)
    )

    print(
        "Window Normals:",
        np.sum(y_test_window == 0)
    )

    print(
        "Window Anomalies:",
        np.sum(y_test_window == 1)
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
        "Normal Windows:",
        np.sum(y_test_window == 0)
    )

    print(
        "Anomaly Windows:",
        np.sum(y_test_window == 1)
    )
