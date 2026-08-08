# Avocado Smash

A keyboard-first Pygame demo for an eFlesh tactile sensor game. The live
classifier consumes four magnetometers with twelve magnetic channels:

```text
[s0x, s0y, s0z, ..., s3x, s3y, s3z]
```

The game derives three ergonomic gesture intents from that XYZ vector:

- `Wrist Up` - flex the wrist upward and hold briefly
- `Spread` - spread and straighten all fingers
- `Fist` - close the whole hand into a fist

This makes the demo feel closer to a real wearable/tactile interface, where some fingers are hard to isolate and multi-finger combinations can be tiring or unreliable.

## Run

```powershell
cd avocado_smash
python -m pip install -r requirements.txt
python main.py
```

After creating `..\calibration_pipeline\profile.json`, enable live gesture
detection while retaining the three-key keyboard fallback:

```powershell
python main.py --sensor-port COM3
```

The game starts while a background worker collects the two-second relaxed
baseline from all four sensors. It then uses ESP32 timestamps to resample a
rolling 360 ms sensor history before applying the causal moving average. The
gesture bars update once per physical frame, while a label must win two
consecutive predictions before it drives a gesture onset action. The sensor
panel reports `live, timestamped` when detection is ready.

Speed options:

```powershell
python main.py --speed 1.5   # faster
python main.py --speed 0.75  # slower
```

## Controls

| Gesture | Key | Why this shape |
| --- | --- | --- |
| Wrist Up | F | Wrist extension held briefly for reliable detection. |
| Spread | J | Sustained extensor motion, intended to oppose a fist response. |
| Fist | K | Sustained whole-hand flexor motion. |

Gesture symbols on falling avocados:

| Symbol | Motion |
| --- | --- |
| `*` | Wrist Up |
| `<>` | Spread |
| `[]` | Fist |

Other keys:

- `Space` or `Enter`: start or restart
- `P`: pause or resume
- `Esc`: quit

## Gameplay

The playfield is a single rhythm lane. Every avocado follows the same vertical
line, and its symbol tells you which of the three motions to perform: wrist up,
finger spread, or fist. Press the matching gesture key when the avocado
overlaps the outlined receptor on the horizontal hit line.

Lives are unlimited, so missed avocados reset the combo but do not end the
session.

The hollow avocado-shaped outline on the hit line shows the exact timing
target, similar to a Taiko receptor. The right panel shows only the three
derived gesture values.

The keyboard simulation remains isolated in `EfleshKeyboardMirror`. With
`--sensor-port`, `LiveGestureClient` reads KNN labels on a background thread,
updates the intent display, and turns new non-rest labels into game actions.
Keyboard gesture keys remain usable as a fallback.
