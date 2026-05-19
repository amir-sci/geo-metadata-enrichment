from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PARSED_DIR, PREPROC_FILE
from .utils import json_to_list, load_dataframe, normalize_text, save_dataframe


def _ensure_list_column(series: pd.Series) -> pd.Series:
    return series.apply(json_to_list)


def preprocess_df(input_path: str | Path | None = None, output_dir: str | Path | None = None) -> pd.DataFrame:
    input_path = Path(input_path or (PARSED_DIR / "canonical_raw.parquet"))
    output_dir = Path(output_dir or PARSED_DIR)
    df = load_dataframe(input_path)

    for col in ["keywords", "place_names", "origin", "topic_category", "contact"]:
        if col in df.columns:
            df[col] = _ensure_list_column(df[col])

    df["title_norm"] = df["title"].apply(normalize_text) if "title" in df.columns else None
    df["abstract_norm"] = df["abstract"].apply(normalize_text) if "abstract" in df.columns else None
    df["purpose_norm"] = df["purpose"].apply(normalize_text) if "purpose" in df.columns else None
    df["keywords_norm"] = df["keywords"].apply(lambda x: [normalize_text(v) for v in x if normalize_text(v)] if isinstance(x, list) else [])
    df["place_norm"] = df["place_names"].apply(lambda x: [normalize_text(v) for v in x if normalize_text(v)] if isinstance(x, list) else [])
    df["text_for_embedding"] = df.apply(
        lambda r: " . ".join([p for p in [r.get("title_norm"), r.get("abstract_norm"), " ".join(r.get("keywords_norm", [])), " ".join(r.get("place_norm", [])), r.get("purpose_norm")] if isinstance(p, str) and p.strip()]),
        axis=1,
    )
    df["has_bbox"] = df["bbox"].apply(lambda x: bool(x) and str(x) != "nan") if "bbox" in df.columns else False
    df["has_title"] = df["title_norm"].notna() if "title_norm" in df.columns else False
    df["has_abstract"] = df["abstract_norm"].notna() if "abstract_norm" in df.columns else False
    df["has_keywords"] = df["keywords_norm"].apply(bool)
    df["has_place"] = df["place_norm"].apply(bool)
    df["dataset_group"] = df["filename"].astype(str).str.split("_").str[0].fillna("unknown")

    save_dataframe(df, output_dir / "canonical_preproc.parquet")
    save_dataframe(df, output_dir / PREPROC_FILE)
    return df
