"""JSON-file-backed model registry (Phase 32).

Every training run's metadata (version, training date, training universe, feature
schema version, validation method, validation/test metrics) is appended to one log
file, so a served model version is discoverable without re-running training. A JSON
file rather than a database table by design: training is a batch/offline step
decoupled from the FastAPI app's SQLite database (src/db/schema.py), and this
project's scale -- one entry per training run -- doesn't need a relational table.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
REGISTRY_FILENAME = "registry.json"
# A second, independent registry log for quantile-regression models (src.models.
# train_quantiles.py) -- kept as a separate file rather than mixed into
# registry.json since a quantile-model entry's shape (model_paths per quantile,
# coverage metric) differs from a classifier entry's, and callers always know
# which kind they want.
QUANTILE_REGISTRY_FILENAME = "quantile_registry.json"


def _registry_path(model_dir: Path, filename: str = REGISTRY_FILENAME) -> Path:
    return model_dir / filename


def register_model(entry: dict, model_dir: Path = DEFAULT_MODEL_DIR, filename: str = REGISTRY_FILENAME) -> None:
    """Appends one entry to the registry log, keyed by model_version (re-registering
    the same version replaces its old entry rather than duplicating it).
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    entries = [e for e in load_all(model_dir, filename) if e["model_version"] != entry["model_version"]]
    entries.append(entry)
    _registry_path(model_dir, filename).write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")


def load_all(model_dir: Path = DEFAULT_MODEL_DIR, filename: str = REGISTRY_FILENAME) -> list[dict]:
    path = _registry_path(model_dir, filename)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest(model_dir: Path = DEFAULT_MODEL_DIR, filename: str = REGISTRY_FILENAME) -> dict | None:
    """Returns the most recently trained registry entry, or None if none exist yet."""
    entries = load_all(model_dir, filename)
    if not entries:
        return None
    return max(entries, key=lambda e: e["training_date"])


def load_version(model_version: str, model_dir: Path = DEFAULT_MODEL_DIR, filename: str = REGISTRY_FILENAME) -> dict:
    for entry in load_all(model_dir, filename):
        if entry["model_version"] == model_version:
            return entry
    raise ValueError(f"No registry entry for model_version={model_version!r}")
