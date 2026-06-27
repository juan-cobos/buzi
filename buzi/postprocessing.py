import numpy as np


class Events:
    """A chainable set of detected intervals over a sampled signal.

    Parameters
    ----------
    intervals : array-like, shape (n_events, 2)
        Half-open ``[start, stop)`` sample-index pairs.
    fs : float
        Sampling rate in Hz, used to express durations and gaps in seconds.
    """

    def __init__(self, intervals, fs: float):
        self.intervals = np.asarray(intervals, dtype=int).reshape(-1, 2)
        self.fs = float(fs)

    @classmethod
    def from_threshold(cls, traces, fs: float, low: float, peak: float) -> "Events":
        """Build events by thresholding one or more detection traces.

        For each trace (row of ``traces``), takes contiguous runs above ``low``
        whose maximum exceeds ``peak``, then unions the runs across traces
        (overlapping intervals are merged). Passing several traces and unioning
        is the Karlsson-style multi-channel detection; a single combined trace
        gives the Kay/Zugaro style.

        Parameters
        ----------
        traces : array-like, shape (n_traces, n_times) or (n_times,)
            Z-scored detection trace(s), in standard-deviation units.
        fs : float
            Sampling rate in Hz.
        low, peak : float
            Interval boundary and required-peak thresholds, in SD.
        """
        traces = np.atleast_2d(np.asarray(traces, dtype=float))
        runs = [_runs_above(tr, low, peak) for tr in traces]
        runs = [r for r in runs if len(r)]
        intervals = np.vstack(runs) if runs else np.empty((0, 2), int)
        return cls(intervals, fs).merge()

    def merge(self, min_interval: float = 0.0) -> "Events":
        """Merge events separated by less than ``min_interval`` seconds.

        With the default of 0 this only fuses touching or overlapping
        intervals (the union used to combine multi-channel detections); a
        positive value also bridges short gaps between distinct events.
        """
        segs = self.intervals
        if len(segs) > 1:
            gap = round(min_interval * self.fs)
            segs = segs[np.argsort(segs[:, 0])]
            merged = [segs[0].copy()]
            for start, stop in segs[1:]:
                if start - merged[-1][1] <= gap:
                    merged[-1][1] = max(merged[-1][1], stop)
                else:
                    merged.append(np.array([start, stop]))
            segs = np.array(merged)
        self.intervals = segs
        return self

    def filter_duration(
        self, min_duration: float = 0.0, max_duration: float | None = None
    ) -> "Events":
        """Keep only events whose duration lies within the given bounds.

        Durations are in seconds. ``max_duration=None`` leaves events
        unbounded above (set e.g. 0.1 to mimic buzcode's 100 ms cap).
        """
        if len(self.intervals):
            dur = (self.intervals[:, 1] - self.intervals[:, 0]) / self.fs
            keep = dur >= min_duration
            if max_duration is not None:
                keep &= dur <= max_duration
            self.intervals = self.intervals[keep]
        return self

    @property
    def starts(self) -> np.ndarray:
        """Event onsets in seconds."""
        return self.intervals[:, 0] / self.fs

    @property
    def stops(self) -> np.ndarray:
        """Event offsets in seconds (last in-bounds sample)."""
        return (self.intervals[:, 1] - 1) / self.fs

    @property
    def durations(self) -> np.ndarray:
        """Event durations in seconds."""
        return self.stops - self.starts

    def __len__(self) -> int:
        return len(self.intervals)

    def __iter__(self):
        return iter(self.intervals)


def _runs_above(trace: np.ndarray, low: float, peak: float) -> np.ndarray:
    """Half-open [start, stop) runs above ``low`` whose max exceeds ``peak``."""
    mask = trace > low
    if mask.size == 0:
        return np.empty((0, 2), int)
    d = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    stops = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        stops = np.r_[stops, mask.size]
    segs = np.column_stack([starts, stops])
    if not len(segs):
        return segs
    return segs[[trace[s:e].max() > peak for s, e in segs]]
