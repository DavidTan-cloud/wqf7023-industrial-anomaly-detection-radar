# src/datasets/test_mfcc.py

import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.datasets.mimii_preprocessor import (
    MIMIIPreprocessor
)

processor = MIMIIPreprocessor()

features = processor.extract_features(
    "src/datasets/raw/MIMII/0_dB_fan/fan/id_00/normal/normal_id_00_00000000.wav"
)

print(features.shape)

PS C:\Users\david\Documents\GitHub\wqf7023-industrial-anomaly-detection-radar> python src\datasets\mfcc_test.py
Traceback (most recent call last):
  File "C:\Users\david\Documents\GitHub\wqf7023-industrial-anomaly-detection-radar\src\datasets\mfcc_test.py", line 21, in <module>
    features = processor.extract_features(
        "src/datasets/raw/MIMII/0_dB_fan/fan/id_00/normal/normal_id_00_00000000.wav"
    )
  File "C:\Users\david\Documents\GitHub\wqf7023-industrial-anomaly-detection-radar\src\datasets\mimii_preprocessor.py", line 20, in extract_features
    audio, sr = librosa.load(
                ~~~~~~~~~~~~^
        filepath,
        ^^^^^^^^^
        sr=self.sample_rate,
        ^^^^^^^^^^^^^^^^^^^^
        mono=True
        ^^^^^^^^^
    )
    ^
  File "C:\Users\david\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\librosa\core\audio.py", line 156, in load
    y, sr_native = __soundfile_load(path, offset, duration, dtype)
                   ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\david\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\librosa\core\audio.py", line 179, in __soundfile_load
    context = sf.SoundFile(path)
  File "C:\Users\david\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\soundfile.py", line 708, in __init__
    self._file = self._open(file, mode_int, closefd)
                 ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\david\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\soundfile.py", line 1296, in _open
    raise LibsndfileError(err, prefix=f"Error opening {self.name!r}: ")
soundfile.LibsndfileError: Error opening 'src/datasets/raw/MIMII/0_dB_fan/fan/id_00/normal/normal_id_00_00000000.wav': System error.