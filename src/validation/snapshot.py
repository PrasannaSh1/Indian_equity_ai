"""Single-analysis-snapshot consistency (website audit, Section 1-2).

Root cause this fixes: independent per-table "latest date" queries let different
data sources (e.g. prices_daily vs model_predictions) silently drift out of sync,
so a dashboard can show a fresh header price next to a forecast/entry-exit computed
from a stale, different date. resolve_snapshot() picks the one shared date every
source actually agrees on, instead of letting each source decide independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class SnapshotConsistency:
    as_of: date
    per_source_dates: dict[str, date] = field(default_factory=dict)
    is_fully_aligned: bool = True
    lagging_sources: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "per_source_dates": {k: v.isoformat() for k, v in self.per_source_dates.items()},
            "is_fully_aligned": self.is_fully_aligned,
            "lagging_sources": self.lagging_sources,
        }


def resolve_snapshot(per_source_dates: dict[str, date | None]) -> SnapshotConsistency:
    """Drops sources with no data (None), then sets as_of to the MIN of the
    remaining sources' MAX dates -- the latest date at which every available
    source actually has data. Sources ahead of as_of are recorded in
    lagging_sources (mapped to how many calendar days behind they'd otherwise be
    reporting), so the caller can show a staleness warning instead of silently
    serving a contradictory mix of dates.
    """
    available = {k: v for k, v in per_source_dates.items() if v is not None}
    if not available:
        raise ValueError("No source has any data; cannot resolve a snapshot date.")

    as_of = min(available.values())
    lagging = {k: (v - as_of).days for k, v in available.items() if v > as_of}
    return SnapshotConsistency(
        as_of=as_of,
        per_source_dates=available,
        is_fully_aligned=not lagging,
        lagging_sources=lagging,
    )


def assert_single_snapshot(dates: dict[str, date]) -> None:
    """Safety-net assertion for callers (e.g. the live orchestrator) that are
    supposed to derive every date from one shared fetch. Raises AssertionError if
    they've ever diverged -- this guards against a future regression reintroducing
    independent per-block fetches, not a currently-known bug.
    """
    unique_dates = set(dates.values())
    if len(unique_dates) > 1:
        raise AssertionError(f"Analysis snapshot dates diverged: {dates}")
