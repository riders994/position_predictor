"""Filesystem helpers: project paths, Parquet caching, content hashing.

Kept dependency-light. ``pandas``/``pyarrow`` are imported lazily inside the functions
that need them so this module (and config loading) can be imported without the full
data-science stack installed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# src/nba_archetypes/utils/io.py -> parents[3] == sports/basketball (this sport's project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
MANIFEST_DIR = DATA_RAW / "_manifests"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve(path: str | Path) -> Path:
    """Resolve ``path`` relative to the project root if it is not absolute."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def write_parquet(df, path: Path):
    """Write a DataFrame to Parquet, creating parent dirs.

    Accepts a polars or pandas frame (sportsdataverse/nba_api loaders may return either). Polars
    frames write natively; for pandas frames, mixed-type ``object`` columns that Arrow refuses to
    serialise are coerced to nullable string on a retry so caching never aborts.
    """
    ensure_dir(path.parent)
    if hasattr(df, "write_parquet"):  # polars frame — native, strictly typed
        df.write_parquet(path)
        return path
    try:
        df.to_parquet(path, index=False)
    except Exception:
        _coerce_mixed_object_columns(df).to_parquet(path, index=False)
    return path


def _coerce_mixed_object_columns(df):
    """Return a copy with mixed-python-type object columns cast to nullable string."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            kinds = {type(v) for v in df[col].dropna()}
            if len(kinds) > 1:
                df[col] = df[col].astype("string")
    return df


def read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file's bytes (for data-version manifests)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
