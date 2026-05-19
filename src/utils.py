from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_text_lower(value: Any) -> str | None:
    text = normalize_text(value)
    return text.lower() if text else None


def split_free_text_list(value: Any) -> list[str]:
    """Split delimited metadata lists while preserving order and removing duplicates."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [str(value)]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if item is None:
            continue
        for part in re.split(r"[;,/|]\s*", str(item)):
            token = normalize_text(part)
            if token and token.lower() not in seen:
                seen.add(token.lower())
                out.append(token)
    return out


def list_to_json(value: Sequence[str] | None) -> str:
    return json.dumps(list(value or []), ensure_ascii=False)


def json_to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if normalize_text(x)]
    if isinstance(value, (tuple, set)):
        return [str(x) for x in value if normalize_text(x)]
    if isinstance(value, float) and math.isnan(value):
        return []
    if not isinstance(value, str):
        value = str(value)
    s = value.strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if normalize_text(x)]
    except Exception:
        pass
    # fallback for legacy repr([...]) strings
    from ast import literal_eval

    try:
        parsed = literal_eval(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if normalize_text(x)]
    except Exception:
        pass
    # final fallback: split on common delimiters
    return split_free_text_list(s)


def save_dataframe(df: pd.DataFrame, base_path: str | Path, *, index: bool = False) -> Path:
    base_path = Path(base_path)
    ensure_dir(base_path.parent)
    suffix = base_path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(base_path, index=index)
            return base_path
        except Exception:
            fallback = base_path.with_suffix(".csv")
            df.to_csv(fallback, index=index)
            return fallback
    if suffix == ".csv":
        df.to_csv(base_path, index=index)
        return base_path
    # default to csv
    csv_path = base_path.with_suffix(".csv")
    df.to_csv(csv_path, index=index)
    return csv_path


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path.with_suffix(".csv"))
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv"))
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    raise FileNotFoundError(path)


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1 - alpha / 2.0))
    return lower, upper


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * ((p * (1 - p) / n) + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def flatten_unique(values: Iterable[Iterable[str] | str | None]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item is None:
            continue
        if isinstance(item, str):
            iterable = [item]
        else:
            iterable = item
        for val in iterable:
            tok = normalize_text(val)
            if tok and tok.lower() not in seen:
                seen.add(tok.lower())
                out.append(tok)
    return out


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default
