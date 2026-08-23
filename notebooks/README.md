# Teaching notebooks

These notebooks are fill-in-the-blank worksheets for the current four-sensor calibration pipeline. They share one fixed, deterministic set of armband-shaped captures, so the filtering lesson and KNN lesson work from the same 50 Hz, 12-channel samples.

1. **01_signal_filtering.ipynb** — select a raw capture, write the causal five-sample moving-average function, then plot its result separately before making the same 19-frame training window and signed-mean/RMS features.
2. **02_knn_gesture_classification.ipynb** — extract the exact eight windows per capture, build the labeled feature matrix, train K=3, inspect a vote, and stabilize a live label.

## Setup on Windows

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\notebooks\requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

Open the notebooks in numeric order. Complete each marked **TODO** or **None** expression before continuing. They import the real functions and constants from **calibration_pipeline/eflesh_calibration**; only serial I/O, user prompts, timestamp resampling, and profile-file writing are omitted.

The fixed captures are teaching data, not a replacement for validation with recordings from the physical armband.
