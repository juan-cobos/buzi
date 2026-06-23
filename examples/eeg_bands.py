"""Compute per-band power from real EEG, end to end.

:func:`buzi.helpers.load_example` only *reads* a short recording from the
PhysioNet EEG Motor Movement/Imagery dataset (small, auto-downloaded by MNE);
everything else -- referencing, preprocessing, band-splitting -- is done by
chaining :class:`buzi.Signal` operations, which is the whole point of the
library.

Run with the ``examples`` dependency group, as a module from the repo root::

    uv run --group examples python -m examples.eeg_bands

The first run downloads ~20 MB of EDF files into MNE's data directory.
"""

from __future__ import annotations

import numpy as np

from buzi import BANDS, Signal
from buzi.helpers import load_example


def preprocess(raw: Signal) -> Signal:
    """Common-average reference, done purely with Signal arithmetic.

    ``combine_channels("mean")`` is the across-channel average ``(1, n_times)``,
    which broadcasts back out when subtracted -- so re-referencing is just
    ``raw - raw.combine_channels("mean")``.
    """
    return raw - raw.combine_channels("mean")


def band_power(signal: Signal) -> dict[str, np.ndarray]:
    """Mean power per channel in each band that fits below Nyquist.

    Power is the time-average of the squared band-passed signal, computed by
    chaining the :class:`~buzi.Signal` pipeline: ``signal.band(name) ** 2``.
    """
    nyquist = signal.fs / 2.0
    powers: dict[str, np.ndarray] = {}
    for name, (_low, high) in BANDS.items():
        if high >= nyquist:
            continue  # e.g. ripple needs fs >> EEG rates
        powers[name] = (signal.band(name) ** 2).numpy().mean(axis=-1)
    return powers


def main() -> None:
    raw = load_example(subject=1, runs=1)
    print(f"loaded {raw.shape[0]} channels x {raw.shape[1]} samples @ {raw.fs:g} Hz\n")

    signal = preprocess(raw)
    powers = band_power(signal)

    # Relative power averaged across channels, as a quick summary.
    mean_power = {name: float(p.mean()) for name, p in powers.items()}
    total = sum(mean_power.values())
    print(f"{'band':<8}{'abs power (uV^2)':>18}{'relative':>12}")
    print("-" * 38)
    for name, p in mean_power.items():
        print(f"{name:<8}{p:>18.3f}{p / total:>11.1%}")


if __name__ == "__main__":
    main()
