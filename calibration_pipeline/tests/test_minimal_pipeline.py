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
    CALIBRATION_EDGE_SAMPLES,
    CALIBRATION_WINDOW_SAMPLES,
    OnsetTracker,
    PREDICTION_STRIDE_SAMPLES,
    build_parser,
    iter_predictions,
    read_calibration_window,
)
from eflesh_calibration.knn import (
    CLASSES,
    FEATURE_COUNT,
    FILTER_SAMPLES,
    K,
    WINDOW_SAMPLES,
    feature,
    load_model,
    make_model,
    predict,
    proximity_scores,
    save_model,
)
from eflesh_calibration.sensor import CHANNEL_COUNT, parse_frame


def frame_line(values: np.ndarray | None = None) -> str:
    if values is None:
        values = np.arange(CHANNEL_COUNT, dtype=float)
    return "FRAME,7,140000," + ",".join(str(value) for value in values)


def test_parse_frame() -> None:
    np.testing.assert_allclose(parse_frame(frame_line()), np.arange(CHANNEL_COUNT))
    assert parse_frame("READY,protocol=FRAME_v1") is None

    with pytest.raises(RuntimeError, match="expected 12"):
        parse_frame("FRAME,1,2,1,2,3")
    with pytest.raises(RuntimeError, match="non-numeric"):
        parse_frame(frame_line().replace(",0.0,", ",bad,"))
    bad = np.arange(CHANNEL_COUNT, dtype=float)
    bad[4] = np.nan
    with pytest.raises(RuntimeError, match="finite"):
        parse_frame(frame_line(bad))


def test_feature_preserves_sign_and_rms_magnitude() -> None:
    baseline = np.arange(CHANNEL_COUNT, dtype=float)
    offsets = np.linspace(-2.0, 2.0, CHANNEL_COUNT)
    window = np.tile(baseline + offsets, (WINDOW_SAMPLES, 1))
    result = feature(window, baseline)
    assert result.shape == (FEATURE_COUNT,)
    np.testing.assert_allclose(result[:CHANNEL_COUNT], offsets)
    np.testing.assert_allclose(result[CHANNEL_COUNT:], np.abs(offsets))


def test_feature_median_filter_rejects_single_sample_spike() -> None:
    baseline = np.zeros(CHANNEL_COUNT)
    window = np.ones((WINDOW_SAMPLES, CHANNEL_COUNT))
    window[WINDOW_SAMPLES // 2] = 1000.0
    np.testing.assert_allclose(feature(window, baseline), np.ones(FEATURE_COUNT))


def test_calibration_capture_trims_both_edges() -> None:
    class FakeSensor:
        def __init__(self) -> None:
            self.index = 0

        def read(self) -> np.ndarray:
            row = np.full(CHANNEL_COUNT, self.index, dtype=float)
            self.index += 1
            return row

    window = read_calibration_window(FakeSensor())
    assert CALIBRATION_CAPTURE_SECONDS == 2
    assert CALIBRATION_CAPTURE_SAMPLES == 2 * WINDOW_SAMPLES
    assert CALIBRATION_WINDOW_SAMPLES == WINDOW_SAMPLES
    assert FILTER_SAMPLES % 2 == 1
    assert window.shape == (CALIBRATION_WINDOW_SAMPLES, CHANNEL_COUNT)
    assert np.all(window[0] == CALIBRATION_EDGE_SAMPLES)
    assert np.all(
        window[-1] == CALIBRATION_CAPTURE_SAMPLES - CALIBRATION_EDGE_SAMPLES - 1
    )


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
        window = np.zeros((WINDOW_SAMPLES, CHANNEL_COUNT))
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


def test_live_predictions_use_rolling_window_at_ten_hz() -> None:
    class FakeSensor:
        def __init__(self) -> None:
            self.read_count = 0

        def read(self) -> np.ndarray:
            self.read_count += 1
            return np.zeros(CHANNEL_COUNT)

    sensor = FakeSensor()
    predictions = list(
        itertools.islice(
            iter_predictions(sensor, training_model(), np.zeros(CHANNEL_COUNT)),
            3,
        )
    )
    assert predictions == ["rest", "rest", "rest"]
    assert PREDICTION_STRIDE_SAMPLES == 5
    assert sensor.read_count == WINDOW_SAMPLES + 2 * PREDICTION_STRIDE_SAMPLES


def test_cli_only_exposes_calibrate_and_live() -> None:
    parser = build_parser()
    assert parser.parse_args(["calibrate", "--port", "COM3"]).command == "calibrate"
    assert parser.parse_args(["live", "--port", "COM3"]).command == "live"
    with pytest.raises(SystemExit):
        parser.parse_args(["validate", "--port", "COM3"])


def test_avocado_sensor_bridge_imports() -> None:
    path = Path(__file__).resolve().parents[2] / "avocado_smash" / "live_sensor.py"
    spec = importlib.util.spec_from_file_location("avocado_live_sensor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.LiveGestureClient is not None
