import os
import glob

from src.datasets.mimii_preprocessor import (
    MIMIIPreprocessor
)


class MIMIILoader:

    def __init__(
        self,
        root_dir
    ):
        self.root_dir = root_dir

        self.preprocessor = (
            MIMIIPreprocessor()
        )

    def load_machine(
        self,
        dataset_name="0_dB_fan",
        machine_id="00"
    ):

        machine_dir = os.path.join(
            self.root_dir,
            dataset_name,
            machine_type,
            f"id_{machine_id}"
        )

        normal_files = sorted(
            glob.glob(
                os.path.join(
                    machine_dir,
                    "normal",
                    "*.wav"
                )
            )
        )

        anomaly_files = sorted(
            glob.glob(
                os.path.join(
                    machine_dir,
                    "abnormal",
                    "*.wav"
                )
            )
        )

        return normal_files, anomaly_files
