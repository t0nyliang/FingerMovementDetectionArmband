from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from eflesh_calibration.app import (
    CALIBRATION_CAPTURE_SAMPLES,
    CALIBRATION_CAPTURE_SECONDS,
    CALIBRATION_CLEAN_SAMPLES,
    CALIBRATION_EDGE_SAMPLES,
    EXAMPLES_PER_CLASS,
    LabelStabilizer,
    LIVE_WINDOW_SECONDS,
    LiveWindowResampler,
    MAX_LIVE_GAP_SECONDS,
    OnsetTracker,
    STABILITY_FRAMES,
    TRAINING_WINDOW_STRIDE_SAMPLES,
    build_parser,
    iter_prediction_results,
    recalibrate,
    read_calibration_windows,
    read_for_duration,
    resample_samples,
)
from eflesh_calibration.knn import (
    CLASSES,
    FEATURE_COUNT,
    FILTER_SAMPLES,
    K,
    RAW_WINDOW_SAMPLES,
    WINDOW_SAMPLES,
    feature,
    load_model,
    make_model,
    moving_average,
    predict,
    proximity_scores,
    save_model,
)
from eflesh_calibration.sensor import (
    CHANNEL_COUNT,
    SensorFrame,
    parse_sensor_frame,
)


def frame_line(values: np.ndarray | None = None) -> str:
    if values is None:
        values = np.arange(CHANNEL_COUNT, dtype=float)
    return "FRAME,7,140000," + ",".join(str(value) for value in values)


def test_parse_sensor_frame() -> None:
    timestamped = parse_sensor_frame(frame_line())
    assert timestamped is not None
    assert timestamped.sequence == 7
    assert timestamped.device_us == 140000
    np.testing.assert_allclose(timestamped.values, np.arange(CHANNEL_COUNT))
    assert parse_sensor_frame("READY,protocol=FRAME_v1") is None

    with pytest.raises(RuntimeError, match="expected 12"):
        parse_sensor_frame("FRAME,1,2,1,2,3")
    with pytest.raises(RuntimeError, match="non-numeric"):
        parse_sensor_frame(frame_line().replace(",0.0,", ",bad,"))
    bad = np.arange(CHANNEL_COUNT, dtype=float)
    bad[4] = np.nan
    with pytest.raises(RuntimeError, match="finite"):
        parse_sensor_frame(frame_line(bad))


def test_feature_preserves_sign_and_rms_magnitude() -> None:
    baseline = np.arange(CHANNEL_COUNT, dtype=float)
    offsets = np.linspace(-2.0, 2.0, CHANNEL_COUNT)
    window = np.tile(baseline + offsets, (RAW_WINDOW_SAMPLES, 1))
    result = feature(window, baseline)
    assert result.shape == (FEATURE_COUNT,)
    np.testing.assert_allclose(result[:CHANNEL_COUNT], offsets)
    np.testing.assert_allclose(result[CHANNEL_COUNT:], np.abs(offsets))


def test_moving_average_is_five_sample_and_causal() -> None:
    samples = np.repeat(
        np.arange(8, dtype=float)[:, np.newaxis],
        CHANNEL_COUNT,
        axis=1,
    )
    result = moving_average(samples)
    assert result.shape == (4, CHANNEL_COUNT)
    np.testing.assert_allclose(result[:, 0], [2.0, 3.0, 4.0, 5.0])

    samples[-1] = 1000.0
    changed = moving_average(samples)
    np.testing.assert_allclose(changed[:-1], result[:-1])
    assert changed[-1, 0] != result[-1, 0]


def test_calibration_capture_extracts_short_windows_from_clean_center() -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()

    class FakeSensor:
        def __init__(self) -> None:
            self.index = 0
            self.discard_count = 0

        def discard_pending(self) -> None:
            self.discard_count += 1

        def read(self) -> np.ndarray:
            row = np.full(CHANNEL_COUNT, self.index, dtype=float)
            self.index += 1
            clock.now += 0.02
            return row

    sensor = FakeSensor()
    windows = read_calibration_windows(sensor, clock)
    assert sensor.discard_count == 1
    assert EXAMPLES_PER_CLASS == 10
    assert CALIBRATION_CAPTURE_SECONDS == 2
    assert CALIBRATION_CAPTURE_SAMPLES == 2 * 50
    assert CALIBRATION_CLEAN_SAMPLES == 50
    assert TRAINING_WINDOW_STRIDE_SAMPLES == 5
    assert len(windows) == 8
    assert all(
        window.shape == (RAW_WINDOW_SAMPLES, CHANNEL_COUNT) for window in windows
    )
    assert sensor.index in (100, 101)


def test_timed_capture_resamples_a_slower_sensor_to_fifty_hz() -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()

    class SlowSensor:
        def __init__(self) -> None:
            self.index = 0

        def discard_pending(self) -> None:
            pass

        def read(self) -> np.ndarray:
            self.index += 1
            clock.now += 0.05
            return np.full(CHANNEL_COUNT, clock.now)

    sensor = SlowSensor()
    capture, times = read_for_duration(sensor, 2.0, clock)
    assert 2.0 <= clock.now < 2.05 + 1e-9
    assert 39 <= len(capture) <= 40
    resampled = resample_samples(
        capture,
        times,
        np.arange(CALIBRATION_CAPTURE_SAMPLES) / 50,
    )
    assert resampled.shape == (CALIBRATION_CAPTURE_SAMPLES, CHANNEL_COUNT)
    assert np.isfinite(resampled).all()


def test_timed_capture_collapses_duplicate_host_timestamps() -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()

    class BufferedSensor:
        def __init__(self) -> None:
            self.index = 0

        def discard_pending(self) -> None:
            pass

        def read(self) -> np.ndarray:
            self.index += 1
            # Two queued reads share each coarse clock tick.
            if self.index % 2 == 0:
                clock.now += 0.02
            return np.full(CHANNEL_COUNT, self.index, dtype=float)

    capture, times = read_for_duration(BufferedSensor(), 0.06, clock)
    assert len(capture) == len(times)
    assert len(times) >= 2
    assert np.all(np.diff(times) > 0)
    assert capture[-1, 0] == 6


def training_model() -> dict:
    features = []
    labels = []
    for class_index, label in enumerate(CLASSES):
        center = np.zeros(FEATURE_COUNT)
        if class_index:
            center[class_index - 1] = class_index * 5.0
            center[CHANNEL_COUNT + class_index - 1] = class_index * 5.0
        for example in range(5):
            features.append(center + example * 0.01)
            labels.append(label)
    return make_model(np.stack(features), labels)


def test_knn_predicts_all_four_classes() -> None:
    model = training_model()
    baseline = np.zeros(CHANNEL_COUNT)
    for class_index, label in enumerate(CLASSES):
        window = np.zeros((RAW_WINDOW_SAMPLES, CHANNEL_COUNT))
        if class_index:
            window[:, class_index - 1] = class_index * 5.0
        assert predict(model, feature(window, baseline)) == label


def test_knn_tie_uses_nearest_order() -> None:
    model = {
        "k": K,
        "features": [
            [0.0] * FEATURE_COUNT,
            [1.0] * FEATURE_COUNT,
            [2.0] * FEATURE_COUNT,
            [10.0] * FEATURE_COUNT,
        ],
        "labels": ["rest", "wrist_up", "spread", "fist"],
        "feature_mean": [0.0] * FEATURE_COUNT,
        "feature_scale": [1.0] * FEATURE_COUNT,
    }
    assert predict(model, np.zeros(FEATURE_COUNT)) == "rest"


def test_profile_round_trip_preserves_predictions(tmp_path: Path) -> None:
    model = training_model()
    path = tmp_path / "profile.json"
    save_model(model, path)
    loaded = load_model(path)
    assert loaded == model
    assert predict(loaded, np.zeros(FEATURE_COUNT)) == "rest"


def test_model_normalizes_features_before_knn_distance() -> None:
    features = []
    labels = []
    for label, small, large in (
        ("rest", 0.0, 0.0),
        ("wrist_up", 1.0, 1000.0),
        ("spread", 2.0, 2000.0),
        ("fist", 3.0, 3000.0),
    ):
        for offset in range(5):
            row = np.zeros(FEATURE_COUNT)
            row[0] = small + offset * 0.01
            row[1] = large + offset
            features.append(row)
            labels.append(label)

    model = make_model(np.stack(features), labels)
    assert len(model["feature_mean"]) == FEATURE_COUNT
    assert len(model["feature_scale"]) == FEATURE_COUNT
    assert all(scale > 0 for scale in model["feature_scale"])


def test_proximity_scores_cover_all_classes_and_sum_to_one() -> None:
    model = training_model()
    scores = proximity_scores(model, np.zeros(FEATURE_COUNT))
    assert set(scores) == set(CLASSES)
    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores["rest"] == max(scores.values())


@pytest.mark.parametrize(
    "value",
    [
        {"profile_version": 4},
        {"format": 2, "features": [[0.0]], "labels": ["rest"], "k": 3},
        "not a profile",
    ],
)
def test_old_or_malformed_profile_requests_recalibration(
    tmp_path: Path, value: object
) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="run calibration again"):
        load_model(path)


def test_gesture_onset_requires_a_change_or_rest_rearm() -> None:
    tracker = OnsetTracker()
    assert tracker.update("wrist_up") == "wrist_up"
    assert tracker.update("wrist_up") is None
    assert tracker.update("fist") == "fist"
    assert tracker.update("wrist_up") is None
    assert tracker.update("rest") is None
    assert tracker.update("wrist_up") == "wrist_up"


def test_label_changes_after_two_matching_predictions() -> None:
    stabilizer = LabelStabilizer()
    assert STABILITY_FRAMES == 2
    assert stabilizer.update("fist") == "rest"
    assert stabilizer.update("fist") == "fist"
    assert stabilizer.update("rest") == "fist"
    assert stabilizer.update("rest") == "rest"


def test_label_candidate_count_resets_when_prediction_changes() -> None:
    stabilizer = LabelStabilizer()
    assert stabilizer.update("spread") == "rest"
    assert stabilizer.update("fist") == "rest"
    assert stabilizer.update("fist") == "fist"
    stabilizer.reset()
    assert stabilizer.update("fist") == "rest"


def sensor_frame(sequence: int, device_us: int, value: float = 0.0) -> SensorFrame:
    return SensorFrame(
        sequence,
        device_us,
        np.full(CHANNEL_COUNT, value, dtype=float),
    )


def test_live_window_is_360_ms_at_measured_sensor_rate() -> None:
    builder = LiveWindowResampler()
    window = None
    period_us = 85_034  # approximately 11.76 Hz
    for index in range(6):
        window, reset = builder.add(
            sensor_frame(index, index * period_us, index * period_us / 1_000_000)
        )
        assert not reset
    assert LIVE_WINDOW_SECONDS == pytest.approx(0.36)
    assert window is not None
    assert window.shape == (RAW_WINDOW_SAMPLES, CHANNEL_COUNT)
    assert np.isfinite(window).all()
    assert window[-1, 0] - window[0, 0] == pytest.approx(0.36)


def test_live_window_handles_uint32_wrap_and_sequence_gap() -> None:
    builder = LiveWindowResampler()
    assert builder.add(sensor_frame(0xFFFFFFFE, 0xFFFFFF00))[0] is None
    window, reset = builder.add(sensor_frame(0xFFFFFFFF, 84_744))
    assert window is None
    assert not reset
    window, reset = builder.add(sensor_frame(1, 169_744))
    assert window is None
    assert not reset
    assert builder.elapsed_seconds == pytest.approx(0.17)


def test_large_live_gap_resets_and_requires_fresh_window() -> None:
    builder = LiveWindowResampler()
    builder.add(sensor_frame(0, 0))
    builder.add(sensor_frame(1, 85_000))
    restarted_at = 85_000 + int((MAX_LIVE_GAP_SECONDS + 0.01) * 1_000_000)
    window, reset = builder.add(
        sensor_frame(2, restarted_at)
    )
    assert reset
    assert window is None
    for offset in range(1, 5):
        window, reset = builder.add(
            sensor_frame(2 + offset, restarted_at + offset * 85_000)
        )
        assert not reset
        assert window is None
    window, reset = builder.add(sensor_frame(7, restarted_at + 5 * 85_000))
    assert not reset
    assert window is not None


def test_device_restart_resets_live_timeline() -> None:
    builder = LiveWindowResampler()
    builder.add(sensor_frame(500, 2_000_000))
    builder.add(sensor_frame(501, 2_085_000))
    window, reset = builder.add(sensor_frame(0, 10_000))
    assert reset
    assert window is None
    assert builder.last_sequence == 0
    assert builder.last_device_us == 10_000
    assert builder.elapsed_seconds == 0.0


def test_shared_resampler_uses_explicit_target_grid() -> None:
    times = np.asarray([0.0, 0.1, 0.2])
    samples = np.repeat(times[:, np.newaxis], CHANNEL_COUNT, axis=1)
    targets = np.asarray([0.05, 0.15])
    result = resample_samples(samples, times, targets)
    np.testing.assert_allclose(result[:, 0], targets)


def test_live_predictions_run_once_per_fresh_timestamped_frame() -> None:
    class FakeSensor:
        def __init__(self) -> None:
            self.read_count = 0

        def read_frame(self) -> SensorFrame:
            sequence = self.read_count
            self.read_count += 1
            return sensor_frame(sequence, sequence * 20_000)

    sensor = FakeSensor()
    predictions = list(
        itertools.islice(
            iter_prediction_results(
                sensor, training_model(), np.zeros(CHANNEL_COUNT)
            ),
            3,
        )
    )
    assert [label for label, _scores in predictions] == ["rest", "rest", "rest"]
    assert WINDOW_SAMPLES == 15
    assert RAW_WINDOW_SAMPLES == 19
    assert sensor.read_count == RAW_WINDOW_SAMPLES + 2


def test_cli_exposes_calibrate_recalibrate_and_live() -> None:
    parser = build_parser()
    assert parser.parse_args(["calibrate", "--port", "COM3"]).command == "calibrate"
    assert parser.parse_args(["live", "--port", "COM3"]).command == "live"
    args = parser.parse_args(
        ["recalibrate", "--port", "COM3", "--gesture", "spread"]
    )
    assert args.command == "recalibrate"
    assert args.gesture == "spread"
    with pytest.raises(SystemExit):
        parser.parse_args(["validate", "--port", "COM3"])


def test_recalibrate_replaces_only_selected_gesture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profile.json"
    original = training_model()
    save_model(original, path)
    replacement = [np.full(FEATURE_COUNT, 99.0), np.full(FEATURE_COUNT, 100.0)]

    class FakeSensor:
        def __init__(self, port: str) -> None:
            assert port == "COM3"

        def __enter__(self) -> "FakeSensor":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr("eflesh_calibration.app.Sensor", FakeSensor)
    monkeypatch.setattr(
        "eflesh_calibration.app.collect_baseline",
        lambda sensor: np.zeros(CHANNEL_COUNT),
    )
    monkeypatch.setattr(
        "eflesh_calibration.app.collect_gesture_features",
        lambda sensor, label, baseline: replacement,
    )

    recalibrate("COM3", path, "spread")
    updated = load_model(path)
    for label in ("rest", "wrist_up", "fist"):
        before = [
            row
            for row, row_label in zip(original["features"], original["labels"])
            if row_label == label
        ]
        after = [
            row
            for row, row_label in zip(updated["features"], updated["labels"])
            if row_label == label
        ]
        assert after == before
    spread_rows = [
        row
        for row, label in zip(updated["features"], updated["labels"])
        if label == "spread"
    ]
    assert spread_rows == [row.tolist() for row in replacement]


def test_avocado_sensor_bridge_imports() -> None:
    path = Path(__file__).resolve().parents[2] / "avocado_smash" / "live_sensor.py"
    spec = importlib.util.spec_from_file_location("avocado_live_sensor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.LiveGestureClient is not None
