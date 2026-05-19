from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import PARSED_DIR, REPORT_DIR, TABLE2_FILE, TABLE3_FILE, TABLE4_FILE
from .utils import ensure_dir, save_dataframe, wilson_ci, json_to_list, normalize_text


def _nonempty_series(series: pd.Series) -> pd.Series:
    def _is_nonempty(v):
        if isinstance(v, list):
            return len([x for x in v if normalize_text(x)]) > 0
        if pd.isna(v):
            return False
        if isinstance(v, str):
            s = v.strip()
            return s not in {"", "[]", "None", "nan", "NaN"}
        return bool(v)
    return series.apply(_is_nonempty)


def build_table2_summary(valid_df: pd.DataFrame) -> pd.DataFrame:
    total = len(valid_df)
    rows = [
        {"Metric": "Total records", "Count": total, "Percent (%)": 100.0},
        {"Metric": "Files with title", "Count": int(valid_df["title"].notna().sum()), "Percent (%)": round(100 * valid_df["title"].notna().mean(), 1)},
        {"Metric": "Files with abstract", "Count": int(valid_df["abstract"].notna().sum()), "Percent (%)": round(100 * valid_df["abstract"].notna().mean(), 1)},
        {"Metric": "Files with keywords", "Count": int(_nonempty_series(valid_df["keywords"]).sum()), "Percent (%)": round(100 * _nonempty_series(valid_df["keywords"]).mean(), 1)},
        {"Metric": "Files with place names", "Count": int(_nonempty_series(valid_df["place_names"]).sum()), "Percent (%)": round(100 * _nonempty_series(valid_df["place_names"]).mean(), 1)},
        {"Metric": "Files with bbox", "Count": int(valid_df["bbox"].notna().sum()), "Percent (%)": round(100 * valid_df["bbox"].notna().mean(), 1)},
        {"Metric": "Files with centroids", "Count": int(valid_df["centroid_lat"].notna().sum()), "Percent (%)": round(100 * valid_df["centroid_lat"].notna().mean(), 1)},
    ]
    return pd.DataFrame(rows)


def build_table3_coverage(long_df: pd.DataFrame, valid_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in ["title", "abstract", "keywords", "place"]:
        if field in {"title", "abstract"}:
            before_count = int(valid_df[field].notna().sum())
        elif field == "keywords":
            before_count = int(_nonempty_series(valid_df["keywords"]).sum()) if "keywords" in valid_df else 0
        else:
            before_count = int(_nonempty_series(valid_df["place_names"]).sum()) if "place_names" in valid_df else 0
        after_present = long_df[(long_df["field"] == field) & ((long_df["before_present"] == True) | (long_df["action"] == "auto-fill"))]
        after_count = int(after_present["filename"].nunique())
        total = len(valid_df)
        before_pct = 100 * before_count / total
        after_pct = 100 * after_count / total
        lo_b, hi_b = wilson_ci(before_count, total)
        lo_a, hi_a = wilson_ci(after_count, total)
        rows.append(
            {
                "Field": field,
                "Before (count)": before_count,
                "Before (%)": round(before_pct, 1),
                "Before 95% CI": f"[{lo_b*100:.1f}, {hi_b*100:.1f}]",
                "After (count)": after_count,
                "After (%)": round(after_pct, 1),
                "After 95% CI": f"[{lo_a*100:.1f}, {hi_a*100:.1f}]",
                "Absolute gain (%)": round(after_pct - before_pct, 1),
            }
        )
    return pd.DataFrame(rows)


def build_table4_examples(long_df: pd.DataFrame, valid_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    rows = []
    for field in ["title", "abstract", "keywords", "place"]:
        subset = long_df[long_df["field"] == field].copy()
        subset = subset.sort_values(["confidence", "sem_support_count", "spatial_support_count"], ascending=False)
        top_examples = []
        for _, r in subset.head(top_n).iterrows():
            top_examples.append(
                {
                    "filename": r["filename"],
                    "predicted_value": r["predicted_value"],
                    "confidence": float(r["confidence"]),
                    "action": r["action"],
                    "top3_candidates": r["top3_candidates"],
                }
            )
        rows.append(
            {
                "Field": field,
                "Newly auto-filled (count)": int((subset["action"] == "auto-fill").sum()),
                "Left as suggestion (count)": int((subset["action"] == "suggest").sum()),
                "Median candidate pool size (approx.)": int(subset[["sem_support_count", "spatial_support_count"]].sum(axis=1).median()),
                "Top-3 example values": json.dumps(top_examples, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def write_all_reports(valid_df: pd.DataFrame, long_df: pd.DataFrame, report_dir: str | Path | None = None) -> dict[str, Path]:
    report_dir = Path(report_dir or REPORT_DIR)
    ensure_dir(report_dir)
    table2 = build_table2_summary(valid_df)
    table3 = build_table3_coverage(long_df, valid_df)
    table4 = build_table4_examples(long_df, valid_df)
    paths = {
        "table2": save_dataframe(table2, report_dir / TABLE2_FILE),
        "table3": save_dataframe(table3, report_dir / TABLE3_FILE),
        "table4": save_dataframe(table4, report_dir / TABLE4_FILE),
    }
    return paths
