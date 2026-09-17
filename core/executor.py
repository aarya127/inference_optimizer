"""Executors: close the loop from recommendation to real execution.

Everything in core/techniques/ only predicts, from calibrated cost models.
An Executor actually runs the recommended configuration through the real
backend and reports MEASURED latency alongside the PREDICTED value from the
same profile's cost model -- so a recommendation is checked against a fresh,
live run, not just against the historical LOOCV score it was calibrated
with. This is the "universal tool" acting, not just advising.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from core.profile import ModelHardwareProfile


@dataclass(frozen=True)
class ExecutionResult:
    n_tokens: int
    predicted_prefill_ms: float
    measured_prefill_ms: float
    error_pct: float
    quant_level_used: Optional[str]
    note: str = ""


class Executor(ABC):
    """One Executor per backend -- knows how to actually load a model and
    run a real prefill for a given profile, not just predict one."""

    @abstractmethod
    def supports(self, profile: ModelHardwareProfile) -> bool: ...

    @abstractmethod
    def run_prefill(
        self,
        profile: ModelHardwareProfile,
        n_tokens: int,
        quant_level: Optional[str] = None,
    ) -> ExecutionResult:
        """Actually run a real prefill of ~n_tokens tokens and return
        measured vs predicted latency. quant_level selects which measured
        GGUF file to load, when the profile has quant_levels."""
