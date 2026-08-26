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

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

from src.models.sisvae import SISVAE

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

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=256,
            shuffle=True
        )

        test_dataset = TensorDataset(
            torch.FloatTensor(X_test)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False
        )

        model = SISVAE(
            input_dim=X_train.shape[-1],
            latent_dim=16
        ).to(device)
        
        print(
            "Model Device:",
            next(model.parameters()).device
        )

        param_count = sum(
            p.numel()
            for p in model.parameters()
        )

        print(
            f"Parameters: {param_count:,}"
        )
        
        def vae_loss(recon_x, x, mu, logvar):
            recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
            return recon_loss + kl_loss

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

        EPOCHS = 20

        train_start = time.time()

        model.train()

        for epoch in range(EPOCHS):

            epoch_loss = 0

            for (batch,) in train_loader:

                batch = batch.to(device)

                optimizer.zero_grad()

                recon, mu, logvar = model(batch)

                loss = vae_loss(recon, batch, mu, logvar)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                elapsed = (time.time() - train_start)
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | "
                    f"Loss={epoch_loss/len(train_loader):.6f} | "
                    f"Elapsed={elapsed:.1f}s"
                )

        training_time = (
            time.time() - train_start
        )

        inference_start = time.time()

        model.eval()

        all_scores = []
 
        with torch.no_grad():

            for (batch,) in test_loader:

                batch = batch.to(device)

                recon, mu, logvar = model(batch)

                recon_error = (
                    (batch - recon) ** 2
                ).mean(dim=(1,2))

                kl_div = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp(),
                    dim=1
                )

                batch_scores = (recon_error + kl_div).detach().cpu().numpy()
                
                all_scores.append(
                    batch_scores
                )

        scores = np.concatenate(
            all_scores
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
            "results/ETT/ett_sisvae_partial.csv",
            index=False
        )
        
        print(f"Completed {dataset}")

        print(
            f"Training Time: "
            f"{training_time:.2f}s"
        )

        del model
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
        "SISVAE produced no results — check earlier errors"
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
    "results/ETT/ett_sisvae_results.csv",
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
