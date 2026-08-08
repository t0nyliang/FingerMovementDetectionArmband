# Four-Sensor Finger Movement Armband

Four MLX90393 sensors connect to an ESP32 through PCA9548 channels 0, 2, 5, and
7. Because each sensor is isolated by the mux, all four can use their default
I2C address of `0x18`. The firmware reads them in that channel order and emits
one timestamped frame at a target rate of 50 Hz:

```text
FRAME,sequence,device_us,s0x,s0y,s0z,s1x,s1y,s1z,s2x,s2y,s2z,s3x,s3y,s3z
```

The mux reads are sequential, so the four measurements are grouped into one
near-synchronous frame rather than captured at exactly the same instant.

## Live plot

Upload `mlx90393_live/mlx90393_live.ino`, close Arduino Serial Monitor, and run:

```powershell
python -m pip install -r .\mlx90393_live\requirements.txt
python .\mlx90393_live\plot_live.py --port COM3
```

The default 2x2 dashboard shows Bx, By, and Bz for all four sensors. To focus on
one panel:

```powershell
python .\mlx90393_live\plot_live.py --port COM3 --sensor 2
```

If a sensor cannot be read, its values are sent as `nan` while the remaining
sensors continue updating. The firmware retries unavailable sensors once per
second. The plot also reports sequence gaps so serial data loss is visible.

To read the BNO085 and all four MLX90393 sensors from one ESP32, upload
`motion_pipeline/firmware/bno085_uart_rvc.ino` instead. It emits the same
`FRAME` packets plus `MOTION` packets; the existing plot ignores the motion
packets.

Legacy `SAMPLE` and `DATA` packets are still accepted, but only Sensor 0 will
contain data when using those formats.

## Calibration and detection

The minimal teaching pipeline consumes all twelve channels in each `FRAME`.
It records ten fresh, wall-clock-timed two-second examples of `rest`,
`wrist_up`, `spread`, and `fist`, discarding serial frames buffered during each
countdown and resampling each capture to 50 Hz.
Each example contributes overlapping 300 ms windows from its clean center.
The windows use a causal five-sample moving average and are reduced to 12
signed mean changes plus 12 RMS magnitudes for K-nearest neighbors (KNN)
lookup.

```powershell
cd .\calibration_pipeline
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m eflesh_calibration calibrate --port COM3
.\.venv\Scripts\python.exe -m eflesh_calibration recalibrate --port COM3 --gesture spread
.\.venv\Scripts\python.exe -m eflesh_calibration live --port COM3
```

All four sensors are required. The live command collects a fresh relaxed
baseline, uses ESP32 timestamps to resample a rolling 360 ms sensor history,
and predicts once per physical frame. A label is published after two
consecutive matching predictions.

See [calibration_pipeline/README.md](calibration_pipeline/README.md) for the
guided capture steps and the intentionally simplified design.
