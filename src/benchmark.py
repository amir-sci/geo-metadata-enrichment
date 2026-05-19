from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .config import BENCHMARK_FILE, EVALUATION_DIR
from .utils import ensure_dir, save_dataframe


@contextmanager
def timer(stage: str, rows: int | None = None, results: list[dict] | None = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if results is not None:
            results.append({"stage": stage, "seconds": elapsed, "rows": rows if rows is not None else ""})


def save_benchmark_report(records: list[dict], report_dir: str | Path | None = None) -> Path:
    report_dir = Path(report_dir or EVALUATION_DIR)
    ensure_dir(report_dir)
    df = pd.DataFrame(records)
    return save_dataframe(df, report_dir / BENCHMARK_FILE)
