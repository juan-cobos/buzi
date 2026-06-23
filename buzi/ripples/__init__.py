"""Ripple / sharp-wave-ripple detection.

Bundles the detection algorithms (:mod:`buzi.ripples.algorithms`) and the
:class:`~buzi.ripples.detector.RippleDetector` that drives them.
"""

from buzi.ripples.algorithms import (
    ALGORITHMS,
    Karlsson,
    Kay,
    RippleAlgorithm,
    Zugaro,
    register_algorithm,
)
from buzi.ripples.detector import RippleDetector, Ripples

__all__ = [
    "RippleDetector",
    "Ripples",
    "RippleAlgorithm",
    "register_algorithm",
    "ALGORITHMS",
    "Kay",
    "Karlsson",
    "Zugaro",
]
