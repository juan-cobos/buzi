"""Convenience data loaders.

These helpers depend on optional third-party packages (e.g. MNE). The imports
are done *inside* the functions so the rest of ``buzi`` stays importable
without them; install the extra only when you actually call a loader::

    uv run --group examples python -c "from buzi.helpers import load_example; load_example()"
"""

from __future__ import annotations

from pathlib import Path

from buzi.signal import Signal

__all__ = ["load_example"]


def load_data(path: str | Path, num_channels: int):
    path = Path(path)

    if path.suffix != ".edf":
        raise ValueError("Only edf is supported for now")

    import mne

    raw = mne.io.read_raw_edf(path)
    fs = raw.info["sfreq"]  # Hz, taken straight from the EDF header

    data, _ = raw[num_channels, :]
    print(f"Loaded data shape: {data.shape}")
    return Signal(data, fs=fs)


def load_example(subject: int = 1, runs: int | list[int] = 1) -> Signal:
    """Load a raw PhysioNet EEGBCI recording as a :class:`Signal`.

    Downloads (and caches) a short EEG run via MNE and wraps the *unfiltered*
    ``(n_channels, n_times)`` array in microvolts in a ``Signal`` carrying the
    sampling rate. MNE is used purely as an EDF reader -- all filtering and
    preprocessing is meant to be done with the chainable ``Signal`` pipeline,
    not here.

    MNE is imported lazily, so it is only required when this function is
    called. Install it with the ``examples`` dependency group.

    Parameters
    ----------
    subject : int
        PhysioNet subject number.
    runs : int or list of int
        Run number(s) to load and concatenate.
    """
    import mne  # optional dependency, imported on demand

    paths = mne.datasets.eegbci.load_data(
        subjects=subject, runs=runs, update_path=True, verbose="ERROR"
    )
    raw = mne.concatenate_raws(
        [mne.io.read_raw_edf(p, preload=True, verbose="ERROR") for p in paths]
    )
    mne.datasets.eegbci.standardize(raw)  # strip trailing dots from names
    raw.pick("eeg")

    data = raw.get_data() * 1e6  # volts -> microvolts
    return Signal(data, raw.info["sfreq"])
