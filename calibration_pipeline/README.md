# Minimal Four-Sensor Calibration

This is a deliberately small teaching pipeline for recognizing `wrist_up`,
`spread`, and `fist` with four MLX90393 sensors. It favors readable code over
production accuracy.

The ESP32 must send 50 Hz frames with twelve XYZ values:

```text
FRAME,sequence,device_us,s0x,s0y,s0z,...,s3x,s3y,s3z
```

## How it works

1. Average two seconds of relaxed sensor data for a baseline.
2. Capture five two-second recordings of `rest`, `wrist_up`, `spread`, and
   `fist`.
3. Trim half a second from each edge and median-filter the clean one-second
   center.
4. After subtracting the baseline, calculate a signed mean and RMS magnitude
   for each of the 12 channels.
5. Save those 20 labeled 24-value feature vectors and their normalization
   statistics to `profile.json`.
6. Standardize live features and classify a rolling one-second window every
   0.1 seconds using the three nearest examples.

There is no resampling, threshold tuning, validation trial, replay, or model
fitting beyond storing examples for KNN.

The implementation has three main files:

- `sensor.py` parses frames and reads the serial port.
- `knn.py` creates RMS features and performs nearest-neighbor voting.
- `app.py` provides calibration, live use, and the command line.

## Setup

Upload `..\mlx90393_live\mlx90393_live.ino` and close Arduino Serial Monitor.
Then install the two runtime dependencies:

```powershell
cd C:\Cody\FingerMovementDetectionArmband\calibration_pipeline
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Calibrate

```powershell
.\.venv\Scripts\python.exe -m eflesh_calibration calibrate --port COM3
```

Keep the armband position fixed. The guide first records a relaxed baseline,
then captures each class in order. Press Enter once before each set and follow
the three-second countdowns. For `wrist_up`, flex your wrist upward after `GO`
and hold it through the two-second recording. The program prints
`capturing for 2 seconds...` while it records. The first and last half-second
are discarded, leaving one second for the complete movement and held
position and making the capture less sensitive to movement timing.
Calibration always saves after all 20 examples.

## Run live

```powershell
.\.venv\Scripts\python.exe -m eflesh_calibration live --port COM3
```

Live mode takes a fresh relaxed baseline and prints median-filtered, normalized
predictions at 10 Hz from a rolling one-second window. `ONSET` is printed when
a new non-rest gesture appears.

Profiles from the older pipeline are intentionally unsupported. Run calibration
again if the profile format error appears.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```
