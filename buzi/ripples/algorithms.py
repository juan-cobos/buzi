from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from buzi.signal import Signal


@runtime_checkable
class RippleAlgorithm(Protocol):
    """Strategy contract. An algorithm turns a filtered LFP
    :class:`~buzi.signal.Signal` into one or more z-scored *detection traces*
    (via a ``Signal`` chain), plus the boundary/peak thresholds used to carve
    candidate intervals out of them."""

    low_threshold: float  # interval boundary, in SD of the detection trace
    peak_threshold: float  # required peak, in SD

    def apply(self, signal: Signal) -> np.ndarray:
        """Filtered LFP ``Signal`` -> (n_traces, n_times) z-scored traces."""

    def peak_index(
        self, signal: Signal, traces: np.ndarray, start: int, stop: int
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

    def peak_index(self, signal, traces, start, stop):
        metric = traces.max(axis=0)
        return start + int(np.argmax(metric[start:stop]))


@register_algorithm("kay")
@dataclass
class Kay(_BaseAlgorithm):
    peak_threshold: float = 2.0
    low_threshold: float = 0.0  # extend candidate boundaries to the mean
    smoothing_sigma: float = 0.004  # s

    def apply(self, signal):
        return (
            signal.envelope()
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

    def apply(self, signal):
        # no combine_channels -> one z-scored trace per channel
        return signal.envelope().filter_gaussian(self.smoothing_sigma).zscore().data


@register_algorithm("zugaro")
@dataclass
class Zugaro(_BaseAlgorithm):
    """buzcode ``bz_FindRipples``: squared signal, boxcar-smoothed, dual
    threshold. Peak is the trough of the filtered LFP, not the envelope max."""

    peak_threshold: float = 5.0
    low_threshold: float = 2.0
    window: int = 11  # boxcar length in samples
    trough_peak: bool = True

    def apply(self, signal):
        return (
            (signal**2)
            .combine_channels("sum")
            .filter_uniform(self.window)
            .zscore()
            .data
        )

    def peak_index(self, signal, traces, start, stop):
        if self.trough_peak:
            return start + int(np.argmin(signal.data[0, start:stop]))
        return super().peak_index(signal, traces, start, stop)
