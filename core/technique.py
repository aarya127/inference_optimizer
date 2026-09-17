"""The Technique abstraction: one interface, many (model, backend) profiles.

This is the concrete answer to "different techniques for different
models": a Technique declares whether it applies to a given
ModelHardwareProfile (core/profile.py) and, when it does, produces a
Recommendation from that profile's REAL calibrated data -- never from a
hardcoded model-specific constant. The same Technique class runs unmodified
against SmolVLM/MLX and Qwen2.5/llama.cpp; only the profile passed in
changes what comes out (or whether it applies at all).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from core.profile import ModelHardwareProfile


@dataclass(frozen=True)
class Recommendation:
    applicable: bool
    choice: Optional[Any]
    estimated_value: Optional[float]
    rationale: str


class Technique(ABC):
    name: str

    @abstractmethod
    def applies_to(self, profile: ModelHardwareProfile) -> bool:
        """Whether this technique has real, measured data to act on for
        this profile -- not whether it could theoretically apply."""

    @abstractmethod
    def recommend(self, profile: ModelHardwareProfile, **kwargs) -> Recommendation:
        """Produce a Recommendation. Callers should check applies_to()
        first; implementations should still return a non-applicable
        Recommendation rather than raising, so callers can handle both
        uniformly."""
