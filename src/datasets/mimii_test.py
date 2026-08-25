import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)

print(PROJECT_ROOT)

from src.datasets.mimii_loader import MIMIILoader

loader = MIMIILoader(
    root_dir="src/datasets/raw/MIMII"
)

normal_files, anomaly_files = loader.load_machine(
    dataset_name="0_dB_fan",
    machine_id="00"
)

print("Normal:", len(normal_files))
print("Anomaly:", len(anomaly_files))
print(normal_files[:3])
