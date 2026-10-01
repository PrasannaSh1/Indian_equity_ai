"""Entry/exit setup validation (website audit, Section 15).

src.entryexit.engine.compute_entry_exit's formula is correct by construction
(stop = close - multiple*atr is always < close for the same row); an observed
"stop above current price" bug is a symptom of a stale/mismatched snapshot
upstream, not a formula defect. This module is the safety net that catches any
setup that *does* end up logically inconsistent (whatever the cause) before it
reaches the UI, and marks it INVALID rather than presenting it as actionable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EntryExitValidation:
    valid: bool
    reason: str | None = None


def _is_missing(value: float) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def validate_long_setup(stop: float, entry_low: float, entry_high: float, target: float) -> EntryExitValidation:
    """Long-only ordering check: stop < entry_low <= entry_high < target.

    A row with no signal (all four values NaN/None, per
    src.entryexit.engine.compute_entry_exit's has_signal masking) is valid by
    definition -- there is nothing to validate when no trade is being suggested.
    """
    values = (stop, entry_low, entry_high, target)
    if all(_is_missing(v) for v in values):
        return EntryExitValidation(valid=True, reason=None)
    if any(_is_missing(v) for v in values):
        return EntryExitValidation(valid=False, reason="Entry/exit setup has a partial (some but not all) signal.")

    if not stop < entry_low:
        return EntryExitValidation(
            valid=False, reason=f"Stop ({stop}) is not below entry_low ({entry_low}) for a long setup."
        )
    if not entry_low <= entry_high:
        return EntryExitValidation(
            valid=False, reason=f"entry_low ({entry_low}) exceeds entry_high ({entry_high})."
        )
    if not entry_high < target:
        return EntryExitValidation(
            valid=False, reason=f"Target ({target}) is not above entry_high ({entry_high}) for a long setup."
        )
    return EntryExitValidation(valid=True, reason=None)
