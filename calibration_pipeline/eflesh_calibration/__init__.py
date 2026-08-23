"""Minimal four-sensor KNN teaching pipeline."""

from .knn import feature, load_model, predict
from .sensor import Sensor

__all__ = ["Sensor", "feature", "load_model", "predict"]
