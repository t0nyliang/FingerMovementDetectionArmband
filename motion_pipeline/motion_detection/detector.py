"""Stateful hysteresis detector for dynamic wrist motion."""

from __future__ import annotations

from .features import MotionFeatures, MotionThresholds


class MotionDetector:
    """Use separate on/off confirmation counts to avoid chatter."""

    def __init__(
        self,
        thresholds: MotionThresholds | None = None,
        on_frames: int = 2,
        off_frames: int = 4,
        off_score: float = 0.65,
    ) -> None:
        if on_frames < 1 or off_frames < 1 or off_score <= 0:
            raise ValueError("detector counts and off_score must be positive")
        self.thresholds = thresholds or MotionThresholds()
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.off_score = off_score
        self.reset()

    def reset(self) -> None:
        self.state = "rest"
        self._motion_count = 0
        self._rest_count = 0

    def update(self, features: MotionFeatures) -> tuple[float, str | None]:
        score = self.thresholds.score(features)
        if self.state == "rest":
            return score, self._update_rest_state(score)
        return score, self._update_motion_state(score)

    @staticmethod
    def _consecutive_count(count: int, condition: bool) -> int:
        """Extend a run when its condition holds, otherwise restart it."""
        return count + 1 if condition else 0

    def _update_rest_state(self, score: float) -> str | None:
        self._rest_count = 0
        self._motion_count = self._consecutive_count(
            self._motion_count, score >= 1.0
        )
        if self._motion_count < self.on_frames:
            return None
        self.state = "motion"
        self._motion_count = 0
        return "motion_onset"

    def _update_motion_state(self, score: float) -> str | None:
        self._motion_count = 0
        self._rest_count = self._consecutive_count(
            self._rest_count, score < self.off_score
        )
        if self._rest_count < self.off_frames:
            return None
        self.state = "rest"
        self._rest_count = 0
        return "motion_offset"
