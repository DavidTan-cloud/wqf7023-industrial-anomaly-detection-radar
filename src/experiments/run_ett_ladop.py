import os
import sys
import time

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

from src.models.ladop import LADOP

from src.evaluation.metrics import evaluate
from src.evaluation.thresholding import percentile_threshold

machines = [
    "fan_id00",
    "fan_id02",
    "fan_id04",
    "fan_id06",
    "pump_id00",
    "pump_id02",
    "pump_id04",
    "pump_id06",
    "slider_id00",
    "slider_id02",
    "slider_id04",
    "slider_id06",
    "valve_id00",
    "valve_id02",
    "valve_id04",
    "valve_id06",
]

results = []

for machine in machines:

    try:

        X_train = np.load(
            f"src/datasets/processed/MIMII/{machine}_X_train.npy"
        )

        X_val = np.load(
            f"src/datasets/processed/MIMII/{machine}_X_val.npy"
        )

        X_test = np.load(
            f"src/datasets/processed/MIMII/{machine}_X_test.npy"
        )

        y_test = np.load(
            f"src/datasets/processed/MIMII/{machine}_y_test.npy"
        )

        print(f"\n{machine}")
        print("Train:", X_train.shape)
        print("Val:", X_val.shape)
        print("Test:", X_test.shape)
        print("Labels:", y_test.shape)

        model = LADOP()
        param_count = 0
        train_start = time.time()
        model.fit(X_train)
        training_time = time.time() - train_start

        inference_start = time.time()
        scores = model.score(X_test)
        inference_time = time.time() - inference_start

        threshold = percentile_threshold(scores, percentile=95)
        preds = (scores > threshold).astype(int)
        metrics = evaluate(y_test, preds, scores)
        metrics["TrainingTime"] = training_time
        metrics["InferenceTime"] = inference_time
        metrics["Parameters"] = param_count
        metrics["Machine"] = machine

        results.append(metrics)
        
        pd.DataFrame(results).to_csv(
            "results/MIMII/mimii_ladop_partial.csv",
            index=False
        )
        
        print(f"Completed {machine}")

    except Exception as e:
        print(
            f"Failed {machine}: {e}"
        )

results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("No LADOP results were generated")

expected_cols = ["Machine","Accuracy","Precision","Recall","F1","AUC","TrainingTime","InferenceTime","Parameters"]

missing = set(expected_cols) - set(results_df.columns)

if missing:
    raise RuntimeError(f"Missing columns in results: {missing}")

results_df = results_df[expected_cols]

results_df.to_csv(
    "results/MIMII/mimii_ladop_results.csv",
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

print(
    "Average Training Time:",
    results_df["TrainingTime"].mean(),
    "seconds"
)

print(
    "Average Inference Time:",
    results_df["InferenceTime"].mean(),
    "seconds"
)

print(
    "Model Parameters:",
    results_df["Parameters"].iloc[0]
)
