"""Background bridge from the minimal KNN pipeline to the Pygame loop."""

from __future__ import annotations

from pathlib import Path
import queue
import sys
import threading


CALIBRATION_ROOT = Path(__file__).resolve().parents[1] / "calibration_pipeline"
if str(CALIBRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_ROOT))

from eflesh_calibration.app import (  # noqa: E402
    OnsetTracker,
    collect_baseline,
    iter_prediction_results,
)
from eflesh_calibration.knn import load_model  # noqa: E402
from eflesh_calibration.sensor import Sensor  # noqa: E402


Prediction = tuple[str, bool, dict[str, float]]


class LiveGestureClient:
    """Read labels on a worker thread so serial I/O never blocks the game."""

    def __init__(self, port: str, profile_path: Path) -> None:
        self.port = port
        self.profile_path = profile_path
        self._outputs: queue.Queue[Prediction] = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status_lock = threading.Lock()
        self._status = "starting"

    @property
    def status(self) -> str:
        with self._status_lock:
            return self._status

    def _set_status(self, value: str) -> None:
        with self._status_lock:
            self._status = value

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="eflesh-live", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def poll(self) -> list[Prediction]:
        outputs = []
        while True:
            try:
                outputs.append(self._outputs.get_nowait())
            except queue.Empty:
                return outputs

    def _publish(self, value: Prediction) -> None:
        try:
            self._outputs.put_nowait(value)
        except queue.Full:
            self._outputs.get_nowait()
            self._outputs.put_nowait(value)

    def _run(self) -> None:
        try:
            model = load_model(self.profile_path)
            with Sensor(self.port) as sensor:
                self._set_status("relax for 2s")
                baseline = collect_baseline(sensor)
                self._set_status("live, timestamped")
                tracker = OnsetTracker()
                for label, scores in iter_prediction_results(sensor, model, baseline):
                    if self._stop.is_set():
                        return
                    onset = tracker.update(label)
                    self._publish((label, onset is not None, scores))
        except Exception as exc:  # keep keyboard fallback alive
            self._set_status(f"sensor error: {exc}")
