# Combined BNO085 + MLX90393 Motion Pipeline

This pipeline detects dynamic wrist/arm motion from the Adafruit BNO085 while
preserving the existing four-MLX90393 finger stream. The combined ESP32
firmware emits both timestamped packet types over the same USB serial port.

The pipeline is deliberately modular:

- `firmware/bno085_uart_rvc.ino` reads the BNO085 and four MLX90393 sensors on
  one ESP32 and emits both timestamped serial protocols.
- `motion_detection/protocol.py` validates and parses that protocol.
- `motion_detection/serial_sensor.py` owns serial I/O.
- `motion_detection/features.py` builds timestamp-resampled windows, estimates a
  relaxed baseline, and computes explainable acceleration/orientation features.
- `motion_detection/detector.py` applies threshold ratios plus hysteresis to
  publish `motion_onset` and `motion_offset` events.
- `motion_detection/app.py` provides calibration, raw recording, and live use.

## Hardware

Use UART-RVC rather than putting the BNO085 on the existing PCA9548 I2C mux.
Adafruit notes that the BNO085 I2C implementation is unreliable with ESP32
boards and I2C multiplexers.

For the firmware's default pin assignment:

| BNO085 | ESP32 |
| --- | --- |
| VIN | 3V3 |
| GND | GND |
| SDA, UART data out | GPIO16 (RX) |
| P0 | 3V3 |

P1 remains low/default. UART-RVC is output-only for this pipeline, so the BNO085
SCL pin and ESP32 TX connection are not used. If GPIO16 is unavailable on your
ESP32 board, change `BNO_RX_PIN` in the firmware and use that pin instead.

Mount the board rigidly on the wrist or back of the hand. Use pitch/roll and
acceleration for motion detection; avoid depending on absolute yaw because the
four finger sensors may use magnets near the BNO085 magnetometer.

## Install

From the repository root:

```powershell
\.venv\Scripts\python.exe -m pip install -r .\motion_pipeline\requirements.txt
```

Install the Arduino libraries `Adafruit BNO08x RVC`, `Adafruit MLX90393`, and
their dependencies, then upload `firmware/bno085_uart_rvc.ino`. This is now the
combined sketch; do not upload the two standalone sketches together.

The USB stream contains both `FRAME,...` and `MOTION,...` lines. The existing
MLX and BNO host parsers ignore the other packet type, so the existing
calibration, live detection, plot, and motion commands can consume the same
port one at a time. A single application that needs both signals concurrently
still needs one shared serial reader that dispatches lines by packet prefix.

## Calibrate and run

Keep the armband still while calibration collects the relaxed orientation and
acceleration baseline:

```powershell
\.venv\Scripts\python.exe -m motion_pipeline.motion_detection calibrate `
  --port COM3 `
  --output .\motion_pipeline\baseline.json
```

Run live detection using that baseline:

```powershell
\.venv\Scripts\python.exe -m motion_pipeline.motion_detection detect `
  --port COM3 `
  --baseline .\motion_pipeline\baseline.json
```

The detector prints records like:

```text
STATE,42,840000,rest,0.12,0.08,1.7,
STATE,43,860000,motion,1.88,1.91,4.2,motion_onset
```

The score is the largest ratio of measured motion to its threshold. A score of
1.0 means that at least one dynamic feature crossed its threshold. Motion must
remain above threshold for two windows, and must remain below the release
threshold for four windows, to avoid chatter.

## Record data for tuning

Record a few seconds of stillness, wrist movement, arm swing, and intentional
gesture motion:

```powershell
\.venv\Scripts\python.exe -m motion_pipeline.motion_detection record `
  --port COM3 `
  --seconds 30 `
  --output .\recordings\bno085_motion.csv
```

The default thresholds in `motion_detection/features.py` are starting points,
not a final calibration. Use the CSV to tune them for the wearer's mounting
position and movement style.

## Current scope and next extension

This pipeline detects dynamic motion, not individual finger positions. The
existing MLX90393 classifier remains responsible for `spread` and `fist`.
The next integration step can consume `motion_onset` as a motion gate or add
the BNO085 window features to a separate multimodal gesture model.

If magnetic interference is severe, migrate the firmware transport to direct
SPI and enable the BNO085 Game Rotation Vector, which uses accelerometer and
gyro data without magnetometer-based heading correction.
