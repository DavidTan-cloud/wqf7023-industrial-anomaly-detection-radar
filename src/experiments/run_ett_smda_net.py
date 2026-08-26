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
import torch
import torch.nn as nn

from src.models.smda_net import SMDANet

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

from src.evaluation.metrics import evaluate
from src.evaluation.thresholding import percentile_threshold

datasets = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2"
]

results = []

for dataset in datasets:

    try:

        X_train = np.load(
            f"src/datasets/processed/ETT/{dataset}_X_train.npy"
        )

        X_val = np.load(
            f"src/datasets/processed/ETT/{dataset}_X_val.npy"
        )

        X_test = np.load(
            f"src/datasets/processed/ETT/{dataset}_X_test.npy"
        )

        y_test = np.load(
            f"src/datasets/processed/ETT/{dataset}_y_test.npy"
        )

        print(f"\n{dataset}")
        print("Train:", X_train.shape)
        print("Val:", X_val.shape)
        print("Test:", X_test.shape)
        print("Labels:", y_test.shape)

        print(
            "Normal Windows:",
            np.sum(y_test == 0)
        )

        print(
            "Anomaly Windows:",
            np.sum(y_test == 1)
        )

        X_train_t = torch.FloatTensor(
            X_train
        ).to(device)

        X_test_t = torch.FloatTensor(
            X_test
        ).to(device)


        model = SMDANet(
            input_dim=X_train.shape[-1]
        ).to(device)

        param_count = sum(
            p.numel()
            for p in model.parameters()
        )
        
        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

        
        EPOCHS = 20

        train_start = time.time()

        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            reconstruction = model(
                X_train_t
            )

            loss = criterion(
                reconstruction,
                X_train_t
            )

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                elapsed = time.time() - train_start
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | "
                    f"Loss={loss.item():.6f} | "
                    f"Elapsed={elapsed:.1f}s"
                )

        training_time = (
            time.time() - train_start
        )

        inference_start = time.time()

        with torch.no_grad():
            reconstruction = model(
                X_test_t
            )
            scores = (
                ((X_test_t - reconstruction) ** 2)
                .mean(dim=(1,2))
                .detach()
                .cpu()
                .numpy()
            )

        inference_time = (
            time.time() - inference_start
        )

        threshold = percentile_threshold(scores, percentile=95)
        preds = (scores > threshold).astype(int)
        metrics = evaluate(y_test, preds, scores)

        metrics["TrainingTime"] = training_time
        metrics["InferenceTime"] = inference_time
        metrics["Parameters"] = param_count

        metrics["Dataset"] = dataset
        results.append(metrics)
        
        pd.DataFrame(results).to_csv(
            "results/ETT/ett_smda_net_partial.csv",
            index=False
        )
        
        print(f"Completed {dataset}")

        del model
        del X_train_t
        del X_test_t
        del reconstruction
        del scores

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:

        print(
            f"Failed {dataset}: {e}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError(
        "SMDA Net produced no results — check earlier errors"
    )

expected_cols = [
    "Dataset", 
    "Accuracy", 
    "Precision",
    "Recall", 
    "F1", 
    "AUC",
    "TrainingTime",
    "InferenceTime",
    "Parameters"
]

results_df = results_df[expected_cols]

results_df.to_csv(
    "results/ETT/ett_smda_net_results.csv",
    index=False
)

results_df.head()

results_df.describe()

print(
    "Average F1:",
    results_df["F1"].mean()
)

print(
    "Average AUC:",
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
    int(results_df["Parameters"].iloc[0])
)
