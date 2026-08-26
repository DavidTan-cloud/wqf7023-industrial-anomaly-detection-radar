import os
import sys

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

from src.datasets.ett_loader import ETTLoader

loader = ETTLoader(
    csv_path="src/datasets/raw/ETT/ETTh1.csv"
)

df = loader.load()

print(df.head())

print(df.shape)

X = loader.get_features()

print(X.shape)
