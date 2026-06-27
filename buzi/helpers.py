from collections.abc import Sequence
from pathlib import Path

import mne

from buzi.signal import Signal


def load_data(path: str | Path, channels: int | Sequence[int]):
    path = Path(path)

    if path.suffix != ".edf":
        raise ValueError("Only edf is supported for now")

    raw = mne.io.read_raw_edf(path)
    fs = raw.info["sfreq"]  # Hz, taken straight from the EDF header

    data, _ = raw[channels, :]
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
