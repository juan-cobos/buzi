"""Ripple / sharp-wave-ripple detection.

:class:`RippleDetector` ties the pipeline together: band-pass the LFP, hand it
to a detection algorithm (see :mod:`buzi.ripples.algorithms`) for a z-scored trace,
threshold into :class:`~buzi.postprocessing.Events`, then locate per-event
peaks. It is dependency-light (NumPy + SciPy) and operates on plain
``(n_channels, n_times)`` arrays so the core stays testable and format-agnostic.

The bundled algorithms (``"Kay"``, ``"Karlsson"``, ``"Zugaro"``) are all
*ripple-power* detectors. The sharp-wave + ripple detector (buzcode
``bz_DetectSWR``, which needs a separate deep channel) is a different input
contract and is intentionally not folded into ``algorithm=``.

Example
-------
>>> det = RippleDetector(algorithm="Kay")
>>> ripples = det.detect(lfp, fs=1250.0)   # lfp is (n_channels, n_times)
>>> ripples.to_dataframe()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from buzi.postprocessing import Events
from buzi.ripples.algorithms import ALGORITHMS, register_algorithm
from buzi.signal import Signal

__all__ = ["RippleDetector", "Ripples", "register_algorithm", "ALGORITHMS"]


@dataclass
class Ripples:
    """Detected events. Times are in seconds."""

    start: np.ndarray  # (n,) event onset
    stop: np.ndarray  # (n,) event offset
    peak: np.ndarray  # (n,) peak/trough time
    peak_zscore: np.ndarray  # (n,) detection-trace value at peak
    algorithm: str
    params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.start.size

    @property
    def duration(self) -> np.ndarray:
        return self.stop - self.start

    def to_dataframe(self):  # pragma: no cover - thin convenience wrapper
        import pandas as pd

        return pd.DataFrame(
            {
                "start": self.start,
                "stop": self.stop,
                "peak": self.peak,
                "duration": self.duration,
                "peak_zscore": self.peak_zscore,
            }
        )


@dataclass
class RippleDetector:
    """Configure once, run on many recordings.

    Threshold defaults come from the chosen algorithm; set ``low_threshold`` /
    ``peak_threshold`` to override. ``exclude`` is an optional callable
    ``(start_s, stop_s) -> keep_mask`` to drop events overlapping movement
    (speed) or EMG artifact -- this keeps the behavioral covariate decoupled
    from the detector, unlike ripple_detection's hard speed requirement.
    """

    algorithm: str = "Kay"
    ripple_band: tuple[float, float] = (150.0, 250.0)
    minimum_duration: float = 0.015  # s
    maximum_duration: float | None = None  # s; e.g. 0.100 to mimic buzcode
    minimum_interval: float = 0.0  # s; merge events closer than this
    exclude: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    prefiltered: bool = False  # skip internal bandpass
    filter_order: int = 4
    # threshold overrides (None -> use algorithm defaults)
    low_threshold: float | None = None
    peak_threshold: float | None = None

    def __post_init__(self):
        try:
            algo_cls = ALGORITHMS[self.algorithm.lower()]
        except KeyError:
            raise ValueError(
                f"unknown algorithm {self.algorithm!r}; "
                f"choose from {sorted(ALGORITHMS)}"
            ) from None
        self._algo = algo_cls()
        if self.low_threshold is not None:
            self._algo.low_threshold = self.low_threshold
        if self.peak_threshold is not None:
            self._algo.peak_threshold = self.peak_threshold

    def detect(self, lfp, fs: float, *, times=None) -> Ripples:
        """Run detection on an LFP array.

        Parameters
        ----------
        lfp : array-like, shape (n_channels, n_times) or (n_times,)
            Raw (or pre-filtered, see ``prefiltered``) LFP. 1-D input is
            promoted to a single channel.
        fs : float
            Sampling rate in Hz.
        times : array-like, optional
            Sample timestamps in seconds; defaults to ``arange(n_times) / fs``.
        """
        lfp = np.atleast_2d(np.asarray(lfp, dtype=float))
        times = np.arange(lfp.shape[-1]) / fs if times is None else np.asarray(times)
        signal = Signal(lfp, fs)
        if not self.prefiltered:
            signal = signal.bandpass(*self.ripple_band, order=self.filter_order)

        traces = self._algo.apply(signal)
        low, peak = self._algo.low_threshold, self._algo.peak_threshold

        events = (
            Events.from_threshold(traces, fs, low, peak)
            .merge(self.minimum_interval)
            .filter_duration(self.minimum_duration, self.maximum_duration)
        )
        segs = events.intervals

        # peak per event
        peaks = np.array(
            [self._algo.peak_index(signal, traces, s, e) for s, e in segs], dtype=int
        )
        peak_z = traces.max(axis=0)[peaks] if len(peaks) else np.empty(0)

        start_s = times[segs[:, 0]]
        stop_s = times[segs[:, 1] - 1]
        peak_s = times[peaks]

        if self.exclude is not None and len(start_s):
            keep = np.asarray(self.exclude(start_s, stop_s), bool)
            start_s, stop_s, peak_s, peak_z = (
                start_s[keep],
                stop_s[keep],
                peak_s[keep],
                peak_z[keep],
            )

        return Ripples(
            start=start_s,
            stop=stop_s,
            peak=peak_s,
            peak_zscore=peak_z,
            algorithm=self.algorithm,
            params={
                "ripple_band": self.ripple_band,
                "low_threshold": low,
                "peak_threshold": peak,
                "minimum_duration": self.minimum_duration,
                "maximum_duration": self.maximum_duration,
                "minimum_interval": self.minimum_interval,
            },
        )
