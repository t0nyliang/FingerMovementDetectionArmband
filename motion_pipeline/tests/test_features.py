from __future__ import annotations

import numpy as np
import pytest

from motion_pipeline.motion_detection.features import (
    MotionBaseline,
    MotionThresholds,
    MotionWindowResampler,
    compute_features,
    estimate_baseline,
    load_baseline,
    save_baseline,
)
from motion_pipeline.motion_detection.protocol import MotionFrame


def make_frame(sequence: int, device_us: int, values: np.ndarray) -> MotionFrame:
    return MotionFrame(sequence, device_us, tuple(values.tolist()))


def test_resampler_produces_a_fixed_280_ms_window() -> None:
    builder = MotionWindowResampler()
    baseline = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 9.8])
    window = None
    for index in range(15):
        window, reset = builder.add(make_frame(index, index * 20_000, baseline))
        assert not reset
    assert window is not None
    assert window.shape == (15, 6)
    np.testing.assert_allclose(window, np.tile(baseline, (15, 1)))


def test_resampler_resets_after_a_large_gap() -> None:
    builder = MotionWindowResampler()
    values = np.zeros(6)
    builder.add(make_frame(0, 0, values))
    _window, reset = builder.add(make_frame(1, 20_000, values))
    assert not reset
    _window, reset = builder.add(make_frame(2, 300_001, values))
    assert reset


def test_rest_features_are_near_zero() -> None:
    baseline = MotionBaseline(np.asarray([5.0, 10.0, 15.0, 0.0, 0.0, 9.8]))
    window = np.tile(baseline.values, (15, 1))
    features = compute_features(window, baseline)
    assert features.acceleration_rms == pytest.approx(0.0)
    assert features.orientation_speed_rms == pytest.approx(0.0)
    assert features.orientation_offset_rms == pytest.approx(0.0)


def test_dynamic_acceleration_and_rotation_are_detectable() -> None:
    baseline = MotionBaseline(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 9.8]))
    window = np.tile(baseline.values, (15, 1))
    window[8:, 0] = np.linspace(0.0, 45.0, 7)
    window[8:, 3] += 3.0
    features = compute_features(window, baseline)
    score = MotionThresholds().score(features)
    assert features.acceleration_peak == pytest.approx(3.0 * np.sqrt(1.0))
    assert features.orientation_speed_peak > 100.0
    assert score > 1.0


def test_static_orientation_offset_does_not_count_as_dynamic_motion() -> None:
    baseline = MotionBaseline(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 9.8]))
    window = np.tile(baseline.values, (15, 1))
    window[:, 1] = 30.0
    features = compute_features(window, baseline)
    assert features.orientation_offset_rms == pytest.approx(30.0)
    assert MotionThresholds().score(features) == pytest.approx(0.0)


def test_baseline_round_trip(tmp_path) -> None:
    baseline = estimate_baseline(
        np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ]
        )
    )
    path = tmp_path / "baseline.json"
    save_baseline(baseline, path)
    loaded = load_baseline(path)
    np.testing.assert_allclose(loaded.values, baseline.values)
