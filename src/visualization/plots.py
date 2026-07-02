import matplotlib.pyplot as plt
import numpy as np

def plot_loss(losses):
    plt.figure(figsize=(8,4))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()

def plot_scores(scores, labels):
    plt.figure(figsize=(12,4))
    plt.plot(scores)
    anomalies = np.where(labels == 1)[0]
    plt.scatter(
        anomalies,
        scores[anomalies],
        color="red"
    )

    plt.title("Anomaly Scores")
    plt.show()
