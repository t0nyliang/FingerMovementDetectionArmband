"""Signed feature extraction and a small, normalized K-nearest-neighbor model."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from .sensor import CHANNEL_COUNT


FORMAT_VERSION = 8
SAMPLE_HZ = 50
WINDOW_SAMPLES = 15
FILTER_SAMPLES = 5
RAW_WINDOW_SAMPLES = WINDOW_SAMPLES + FILTER_SAMPLES - 1
FEATURE_COUNT = 2 * CHANNEL_COUNT
K = 3
CLASSES = ("rest", "wrist_up", "spread", "fist")
RECALIBRATE = "profile format changed; run calibration again"


def moving_average(samples: np.ndarray) -> np.ndarray:
    """Return causal five-sample averages without padding or future samples."""
    samples = np.asarray(samples, dtype=float)
    if (
        samples.ndim != 2
        or samples.shape[0] < FILTER_SAMPLES
        or samples.shape[1] != CHANNEL_COUNT
    ):
        raise ValueError(
            f"samples must contain at least {FILTER_SAMPLES} rows of 12 sensor values"
        )
    if not np.isfinite(samples).all():
        raise ValueError("moving-average input must be finite")
    neighborhoods = np.lib.stride_tricks.sliding_window_view(
        samples, FILTER_SAMPLES, axis=0
    )
    return np.mean(neighborhoods, axis=-1)


def feature(raw_window: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Smooth 19 raw frames, then return 15-frame signed mean and RMS features."""
    raw_window = np.asarray(raw_window, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if raw_window.shape != (RAW_WINDOW_SAMPLES, CHANNEL_COUNT):
        raise ValueError("window must contain rows of 12 sensor values")
    if baseline.shape != (CHANNEL_COUNT,):
        raise ValueError("baseline must contain 12 sensor values")
    if not np.isfinite(raw_window).all() or not np.isfinite(baseline).all():
        raise ValueError("feature input must be finite")
    window = moving_average(raw_window)
    delta = window - baseline
    signed_mean = np.mean(delta, axis=0)
    rms = np.sqrt(np.mean(np.square(delta), axis=0))
    return np.concatenate((signed_mean, rms))


def make_model(features: np.ndarray, labels: list[str]) -> dict[str, Any]:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
        raise ValueError("training features must have shape (N, 24)")
    if len(features) != len(labels) or set(labels) != set(CLASSES):
        raise ValueError(
            "training labels must include rest, wrist_up, spread, and fist"
        )
    if not np.isfinite(features).all():
        raise ValueError("training features must be finite")

    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    feature_scale = np.where(feature_scale > 1e-9, feature_scale, 1.0)
    return {
        "format": FORMAT_VERSION,
        "sample_hz": SAMPLE_HZ,
        "window_samples": WINDOW_SAMPLES,
        "raw_window_samples": RAW_WINDOW_SAMPLES,
        "filter_samples": FILTER_SAMPLES,
        "filter_kind": "causal_moving_average",
        "feature_count": FEATURE_COUNT,
        "feature_layout": "signed_mean_then_rms",
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "k": K,
        "features": features.tolist(),
        "labels": list(labels),
    }


def predict(model: dict[str, Any], sample: np.ndarray) -> str:
    """Standardize features, then predict with deterministic KNN voting."""
    distances = normalized_distances(model, sample)
    nearest = np.argsort(distances, kind="stable")[: int(model["k"])]
    nearest_labels = [model["labels"][int(index)] for index in nearest]
    counts = Counter(nearest_labels)
    best_count = max(counts.values())
    return next(label for label in nearest_labels if counts[label] == best_count)


def normalized_distances(model: dict[str, Any], sample: np.ndarray) -> np.ndarray:
    """Return distances after applying the model's saved standardization."""
    features = np.asarray(model["features"], dtype=float)
    sample = np.asarray(sample, dtype=float)
    feature_mean = np.asarray(model["feature_mean"], dtype=float)
    feature_scale = np.asarray(model["feature_scale"], dtype=float)
    normalized_features = (features - feature_mean) / feature_scale
    normalized_sample = (sample - feature_mean) / feature_scale
    return np.linalg.norm(normalized_features - normalized_sample, axis=1)


def proximity_scores(model: dict[str, Any], sample: np.ndarray) -> dict[str, float]:
    """Return display-only class proximity values that sum to one."""
    distances = normalized_distances(model, sample)
    closest_by_class = np.asarray(
        [
            min(
                distance
                for distance, label in zip(distances, model["labels"])
                if label == class_label
            )
            for class_label in CLASSES
        ],
        dtype=float,
    )
    relative = closest_by_class - np.min(closest_by_class)
    weights = np.exp(-relative)
    weights /= np.sum(weights)
    return {
        label: float(weight)
        for label, weight in zip(CLASSES, weights)
    }


def save_model(model: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")


def load_model(path: Path) -> dict[str, Any]:
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
        features = np.asarray(model["features"], dtype=float)
        feature_mean = np.asarray(model["feature_mean"], dtype=float)
        feature_scale = np.asarray(model["feature_scale"], dtype=float)
        labels = list(model["labels"])
        valid = (
            model.get("format") == FORMAT_VERSION
            and model.get("sample_hz") == SAMPLE_HZ
            and model.get("window_samples") == WINDOW_SAMPLES
            and model.get("raw_window_samples") == RAW_WINDOW_SAMPLES
            and model.get("filter_samples") == FILTER_SAMPLES
            and model.get("filter_kind") == "causal_moving_average"
            and model.get("feature_count") == FEATURE_COUNT
            and model.get("feature_layout") == "signed_mean_then_rms"
            and model.get("k") == K
            and features.ndim == 2
            and features.shape[1] == FEATURE_COUNT
            and feature_mean.shape == (FEATURE_COUNT,)
            and feature_scale.shape == (FEATURE_COUNT,)
            and len(features) == len(labels)
            and set(labels) == set(CLASSES)
            and np.isfinite(features).all()
            and np.isfinite(feature_mean).all()
            and np.isfinite(feature_scale).all()
            and np.all(feature_scale > 0)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(RECALIBRATE) from exc
    if not valid:
        raise ValueError(RECALIBRATE)
    return model
