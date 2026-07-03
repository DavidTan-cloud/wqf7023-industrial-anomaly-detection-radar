import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

from src.datasets.smap_loader import SMAPLoader
from src.datasets.label_builder import build_labels

from src.preprocessing.windowing import create_windows
from src.preprocessing.label_windowing import create_window_labels
from src.preprocessing.normalization import DataNormalizer

from src.models.ladop import LADOP

from src.evaluation.metrics import evaluate
from src.evaluation.thresholding import percentile_threshold

loader = SMAPLoader(
    train_dir="src/datasets/raw/SMAP/train",
    test_dir="src/datasets/raw/SMAP/test",
    labels_file="src/datasets/raw/SMAP/labeled_anomalies.csv"
)

metadata = loader.load_metadata()

channels = loader.get_channels()

#Test Mode
#channels = channels[:3]

results = []

WINDOW_SIZE = 100

for channel in channels:

    try:
        channel_id = channel.replace(
            ".npy",
            ""
        )

        row = metadata[
            metadata["chan_id"]
            == channel_id
        ].iloc[0]

        train, test = loader.load_channel(
            channel
        )

        if train.ndim == 1:
            train = train.reshape(-1,1)
            
        if test.ndim == 1:
            test = test.reshape(-1, 1)

        anomaly_sequence = row[
            "anomaly_sequences"
        ]

        labels = build_labels(
            anomaly_sequence,
            len(test)
        )
        
        if len(labels) != len(test):
            raise ValueError(
                f"Label length mismatch for {channel}"
            )

        scaler = DataNormalizer()

        train = scaler.fit_transform(train)
        test = scaler.transform(test)

        X_train = create_windows(
            train,
            WINDOW_SIZE
        )

        X_test = create_windows(
            test,
            WINDOW_SIZE
        )

        y_test = create_window_labels(
            labels,
            WINDOW_SIZE
        )

        model = LADOP()
        model.fit(X_train)
        scores = model.score(X_test)

        threshold = percentile_threshold(scores, percentile=95)
        preds = (scores > threshold).astype(int)
        metrics = evaluate(y_test, preds, scores)

        metrics["Channel"] = channel_id
        results.append(metrics)
        
        pd.DataFrame(results).to_csv(
            "results/smap_ladop_partial.csv",
            index=False
        )
        
        print(f"Completed {channel_id}")

    except Exception as e:
        print(channel, e)


results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("No LADOP results were generated")

expected_cols = ["Channel","Accuracy","Precision","Recall","F1","AUC"]

missing = set(expected_cols) - set(results_df.columns)

if missing:
    raise RuntimeError(f"Missing columns in results: {missing}")

results_df = results_df[expected_cols]

results_df.to_csv(
    "smap_ladop_results.csv",
    index=False
)

results_df.head()

results_df.describe()

print(
    "LADOP Average F1:",
    results_df["F1"].mean()
)

print(
    "LADOP Average AUC:",
    results_df["AUC"].mean()
)
