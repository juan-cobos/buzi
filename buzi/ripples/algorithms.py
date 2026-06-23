"""Ripple-band detection algorithms.

Each algorithm is a strategy that turns filtered LFP into one or more z-scored
*detection traces* (via a :class:`~buzi.signal.Signal` chain) and exposes the
boundary/peak thresholds used to carve candidate intervals out of them.
Algorithms are registered by name and looked up by
:class:`~buzi.ripples.detector.RippleDetector`.

Three ripple-power detectors ship by default:

* ``"Kay"``      -- Kay et al. 2016. One consensus envelope across all
                    channels, single z-score threshold.
* ``"Karlsson"`` -- Karlsson & Frank 2009. Per-channel z-score, union of
                    per-channel detections.
* ``"Zugaro"``   -- the buzcode ``bz_FindRipples`` algorithm (squared signal,
                    boxcar smoothing, dual 2/5-SD threshold, trough peak).

Add your own with the :func:`register_algorithm` decorator; it then becomes
selectable via ``RippleDetector(algorithm=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from buzi.signal import Signal

__all__ = [
    "RippleAlgorithm",
    "register_algorithm",
    "ALGORITHMS",
    "Kay",
    "Karlsson",
    "Zugaro",
]


@runtime_checkable
class RippleAlgorithm(Protocol):
    """Strategy contract. An algorithm turns filtered LFP into one or more
    z-scored *detection traces* (via a :class:`~buzi.signal.Signal` chain),
    plus the boundary/peak thresholds used to carve candidate intervals out of
    them."""

    low_threshold: float  # interval boundary, in SD of the detection trace
    peak_threshold: float  # required peak, in SD

    def transform(self, filtered: np.ndarray, fs: float) -> np.ndarray:
        """Filtered LFP -> (n_traces, n_times) z-scored detection traces."""

    def peak_index(
        self, filtered: np.ndarray, traces: np.ndarray, start: int, stop: int
    ) -> int:
        """Sample index of the event peak within [start, stop)."""


ALGORITHMS: dict[str, type] = {}


def register_algorithm(name: str) -> Callable[[type], type]:
    """Class decorator to add an algorithm under ``name`` (case-insensitive)."""

    def deco(cls: type) -> type:
        ALGORITHMS[name.lower()] = cls
        return cls

    return deco


class _BaseAlgorithm:
    """Default peak finder: argmax of the strongest detection trace."""

    low_threshold: float
    peak_threshold: float

    def peak_index(self, filtered, traces, start, stop):
        metric = traces.max(axis=0)
        return start + int(np.argmax(metric[start:stop]))


@register_algorithm("kay")
@dataclass
class Kay(_BaseAlgorithm):
    peak_threshold: float = 2.0
    low_threshold: float = 0.0  # extend candidate boundaries to the mean
    smoothing_sigma: float = 0.004  # s

    def transform(self, filtered, fs):
        return (
            Signal(filtered, fs)
            .envelope()
            .combine_channels("l2")
            .filter_gaussian(self.smoothing_sigma)
            .zscore()
            .data
        )


@register_algorithm("karlsson")
@dataclass
class Karlsson(_BaseAlgorithm):
    peak_threshold: float = 2.0
    low_threshold: float = 0.0
    smoothing_sigma: float = 0.004  # s

    def transform(self, filtered, fs):
        # no combine_channels -> one z-scored trace per channel
        return (
            Signal(filtered, fs)
            .envelope()
            .filter_gaussian(self.smoothing_sigma)
            .zscore()
            .data
        )


@register_algorithm("zugaro")
@dataclass
class Zugaro(_BaseAlgorithm):
    """buzcode ``bz_FindRipples``: squared signal, boxcar-smoothed, dual
    threshold. Peak is the trough of the filtered LFP, not the envelope max."""

    peak_threshold: float = 5.0
    low_threshold: float = 2.0
    window: int = 11  # boxcar length in samples
    trough_peak: bool = True

    def transform(self, filtered, fs):
        return (
            (Signal(filtered, fs) ** 2)
            .combine_channels("sum")
            .filter_uniform(self.window)
            .zscore()
            .data
        )

    def peak_index(self, filtered, traces, start, stop):
        if self.trough_peak:
            return start + int(np.argmin(filtered[0, start:stop]))
        return super().peak_index(filtered, traces, start, stop)
