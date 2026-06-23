"""Functional array type for neural signals.

:class:`Signal` wraps an LFP array together with its sampling rate and exposes
the signal-conditioning steps as functional operations, in the spirit of a
tinygrad ``Tensor``: every operation returns a *new* ``Signal`` rather than
mutating in place, so transforms compose as a pure pipeline::

    trace = (
        Signal(lfp, fs)
        .bandpass(150, 250)
        .envelope()
        .filter_gaussian(0.004)
        .zscore()
    )

Arithmetic operators (``+ - * / **``, ``abs``, unary ``-``) broadcast over the
underlying array and return ``Signal`` objects too, so power, ratios and the
like read as ordinary math while staying in the pipeline::

    power = Signal(lfp, fs).bandpass(150, 250) ** 2

The current array is always available as :attr:`data`, shape
``(n_channels, n_times)``; steps operate along the time axis (``axis=-1``) and
either preserve the channel dimension or, for :meth:`combine_channels`,
collapse it to a single consensus channel. The same building blocks back the
detection algorithms in :mod:`buzi.ripples.algorithms`.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt, welch

__all__ = ["Signal", "BANDS", "DEFAULT_ORDER"]

# Default Butterworth order for the band-pass filters.
DEFAULT_ORDER = 4

# Canonical LFP frequency bands, in Hz. Edges follow the conventional EEG
# partition (delta/theta/alpha/beta/gamma) plus the hippocampal ripple band.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 100.0),
    "ripple": (150.0, 250.0),
}


class Signal:
    """An immutable, chainable array over a sampled signal.

    Parameters
    ----------
    data : array-like, shape (n_channels, n_times) or (n_times,)
        Signal samples. 1-D input is promoted to a single channel.
    fs : float
        Sampling rate in Hz, used by the time-aware steps.

    The current array is exposed as :attr:`data`. Every operation returns a new
    ``Signal`` (sharing ``fs``); the receiver is never modified.
    """

    def __init__(self, data, fs: float):
        self.data = np.atleast_2d(np.asarray(data, dtype=float))
        self.fs = float(fs)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def times(self) -> np.ndarray:
        """Sample times in seconds, derived from ``fs`` (not stored)."""
        return np.arange(self.data.shape[-1]) / self.fs

    def numpy(self) -> np.ndarray:
        """Return the underlying array (alias of :attr:`data`)."""
        return self.data

    def __getitem__(self, index) -> "Signal":
        """Index the underlying array, returning a new ``Signal``.

        Accepts any NumPy index on the ``(n_channels, n_times)`` array --
        integers, slices, fancy/boolean masks, or a ``(channels, times)``
        tuple -- so selection stays in the pipeline::

            sig[0]            # first channel
            sig[[0, 2]]       # a subset of channels
            sig[:, 100:200]   # a time window across all channels

        The result is promoted back to ``(n_channels, n_times)`` and keeps
        ``fs``. Time slicing resets the time origin to zero, since ``fs`` is
        preserved but no absolute offset is stored.
        """
        return Signal(self.data[index], self.fs)

    def __len__(self) -> int:
        """Number of channels."""
        return self.data.shape[0]

    def __repr__(self) -> str:
        return f"Signal(shape={self.data.shape}, fs={self.fs})"

    @staticmethod
    def _unwrap(other):
        return other.data if isinstance(other, Signal) else other

    # -- arithmetic (functional, broadcasting) ----------------------------
    def __add__(self, other) -> "Signal":
        return Signal(self.data + self._unwrap(other), self.fs)

    __radd__ = __add__

    def __sub__(self, other) -> "Signal":
        return Signal(self.data - self._unwrap(other), self.fs)

    def __rsub__(self, other) -> "Signal":
        return Signal(self._unwrap(other) - self.data, self.fs)

    def __mul__(self, other) -> "Signal":
        return Signal(self.data * self._unwrap(other), self.fs)

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Signal":
        return Signal(self.data / self._unwrap(other), self.fs)

    def __rtruediv__(self, other) -> "Signal":
        return Signal(self._unwrap(other) / self.data, self.fs)

    def __pow__(self, other) -> "Signal":
        return Signal(self.data ** self._unwrap(other), self.fs)

    def __neg__(self) -> "Signal":
        return Signal(-self.data, self.fs)

    def __abs__(self) -> "Signal":
        return Signal(np.abs(self.data), self.fs)

    # -- signal conditioning ----------------------------------------------
    def bandpass(
        self, l_freq: float, h_freq: float, order: int = DEFAULT_ORDER
    ) -> "Signal":
        """Zero-phase Butterworth band-pass to ``[l_freq, h_freq]`` Hz.

        Uses second-order sections with forward-backward filtering
        (:func:`scipy.signal.sosfiltfilt`), so there is no phase distortion;
        the effective order is twice ``order``.
        """
        sos = butter(order, [l_freq, h_freq], btype="band", fs=self.fs, output="sos")
        return Signal(sosfiltfilt(sos, self.data, axis=-1), self.fs)

    def band(self, name: str, order: int = DEFAULT_ORDER) -> "Signal":
        """Band-pass to a named canonical band (see :data:`BANDS`).

        ``name`` is one of ``"delta"``, ``"theta"``, ``"alpha"``, ``"beta"``,
        ``"gamma"``, ``"ripple"``; the per-band convenience methods
        (:meth:`theta`, :meth:`gamma`, ...) are thin wrappers around this.
        """
        try:
            l_freq, h_freq = BANDS[name]
        except KeyError:
            raise ValueError(
                f"unknown band {name!r}; choose from {sorted(BANDS)}"
            ) from None
        return self.bandpass(l_freq, h_freq, order=order)

    def delta(self, order: int = DEFAULT_ORDER) -> "Signal":
        """Band-pass to the delta band."""
        return self.band("delta", order)

    def theta(self, order: int = DEFAULT_ORDER) -> "Signal":
        """Band-pass to the theta band."""
        return self.band("theta", order)

    def alpha(self, order: int = DEFAULT_ORDER) -> "Signal":
        """Band-pass to the alpha band."""
        return self.band("alpha", order)

    def beta(self, order: int = DEFAULT_ORDER) -> "Signal":
        """Band-pass to the beta band."""
        return self.band("beta", order)

    def gamma(self, order: int = DEFAULT_ORDER) -> "Signal":
        """Band-pass to the gamma band."""
        return self.band("gamma", order)

    def ripple(self, order: int = DEFAULT_ORDER) -> "Signal":
        """Band-pass to the ripple band."""
        return self.band("ripple", order)

    def envelope(self) -> "Signal":
        """Amplitude envelope ``|hilbert(x)|`` of each channel.

        The smooth outline of an oscillation's magnitude. Apply to band-passed
        data; a broadband signal yields a meaningless envelope.
        """
        return Signal(np.abs(hilbert(self.data, axis=-1)), self.fs)

    def filter_gaussian(self, sigma: float = 0.004) -> "Signal":
        """Gaussian smoothing along the time axis.

        ``sigma`` is the kernel standard deviation in seconds (converted to
        samples via ``fs``). This is the Kay/Karlsson envelope smoothing; 4 ms
        is their convention.
        """
        out = gaussian_filter1d(
            self.data, sigma=sigma * self.fs, axis=-1, mode="nearest"
        )
        return Signal(out, self.fs)

    def filter_uniform(self, window: int = 11) -> "Signal":
        """Moving-average (boxcar) smoothing along the time axis.

        ``window`` is the boxcar length in samples. This is the buzcode
        ``bz_FindRipples``/Zugaro envelope smoothing.
        """
        out = uniform_filter1d(self.data, size=int(window), axis=-1, mode="nearest")
        return Signal(out, self.fs)

    def zscore(self, axis: int = -1) -> "Signal":
        """Standardize each channel to zero mean and unit standard deviation.

        Computed over the whole time axis. Zero-variance channels pass through
        unscaled to avoid division by zero. The result is in units of standard
        deviations, which is what the detector thresholds operate on.
        """
        mu = self.data.mean(axis=axis, keepdims=True)
        sd = self.data.std(axis=axis, keepdims=True)
        return Signal((self.data - mu) / np.where(sd == 0, 1.0, sd), self.fs)

    def combine_channels(self, method: str = "l2") -> "Signal":
        """Collapse all channels into a single consensus trace.

        Reduces ``(n_channels, n_times)`` to ``(1, n_times)``. Use after
        :meth:`envelope` or squaring to pool ripple power across electrodes
        (the Kay-style consensus); omit it to keep per-channel traces (the
        Karlsson style).

        Parameters
        ----------
        method : {"l2", "sum", "mean", "max"}
            Combination at each time point. ``"l2"`` is ``sqrt(sum(x**2))``
            (Kay's combined envelope); ``"sum"`` matches buzcode's summed
            squared signal.
        """
        x = self.data
        if method == "l2":
            out = np.sqrt(np.sum(x**2, axis=0))
        elif method == "sum":
            out = np.sum(x, axis=0)
        elif method == "mean":
            out = np.mean(x, axis=0)
        elif method == "max":
            out = np.max(x, axis=0)
        else:
            raise ValueError(
                f"unknown method {method!r}; choose from 'l2', 'sum', 'mean', 'max'"
            )
        return Signal(out[None, :], self.fs)

    # -- reductions out of the signal domain ------------------------------
    def psd(self, nperseg: int = 2048):
        """Welch power spectral density of each channel.

        A terminal operation (like reading :attr:`data`): it leaves the signal
        domain, so it returns plain arrays rather than a :class:`Signal`.
        ``freqs`` has shape ``(n_freqs,)`` and ``psd`` has shape
        ``(n_channels, n_freqs)``.
        """
        freqs, psd = welch(self.data, self.fs, nperseg=nperseg)
        return freqs, psd
