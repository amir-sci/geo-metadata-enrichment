from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CANDIDATE_DIR,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_TAU,
    LONG_SUGGESTIONS_FILE,
    PARSED_DIR,
    SPACY_FILE,
    TARGET_FIELDS,
    EXCLUDED_AUTO_FIELDS,
    WIDE_SUGGESTIONS_FILE,
)
from .utils import json_to_list, load_dataframe, normalize_text, save_dataframe


EXCLUDED_FIELDS = set(EXCLUDED_AUTO_FIELDS)


def _candidate_value_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_text(v) for v in value if normalize_text(v)]
    if isinstance(value, str):
        return json_to_list(value)
    return []


def _score_components(sem_score: float, spatial_score: float, freq_score: float, alpha: float, beta: float, gamma: float) -> float:
    return float(alpha * sem_score + beta * spatial_score + gamma * freq_score)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _extract_neighbor_support(
    df: pd.DataFrame,
    candidate_graph: pd.DataFrame,
    spacy_df: pd.DataFrame,
    record_idx: int,
    field: str,
    current_spacy_pool: list[str] | None = None,
) -> dict[str, Any]:
    row = candidate_graph.iloc[record_idx]
    sem_idx = json_to_list(row.get("semantic_neighbor_indices"))
    spatial_idx = json_to_list(row.get("spatial_neighbor_indices"))
    sem_idx = [int(float(x)) for x in sem_idx]
    spatial_idx = [int(float(x)) for x in spatial_idx]

    semantic_pool: list[str] = []
    spatial_pool: list[str] = []
    spacy_pool: list[str] = []

    def add_values(target_pool: list[str], neighbor_indices: list[int], source_field: str):
        for j in neighbor_indices:
            if j < 0 or j >= len(df):
                continue
            value = df.iloc[j].get(source_field)
            if source_field in {"keywords_norm", "place_norm"}:
                items = _candidate_value_list(value)
            else:
                items = [normalize_text(value)] if normalize_text(value) else []
            target_pool.extend([x for x in items if x])

    if field == "keywords":
        add_values(semantic_pool, sem_idx, "keywords_norm")
        add_values(spatial_pool, spatial_idx, "keywords_norm")
        if current_spacy_pool is None:
            spacy_row = spacy_df.iloc[record_idx]
            spacy_pool.extend(json_to_list(spacy_row.get("spacy_noun_chunks")))
            spacy_pool.extend(json_to_list(spacy_row.get("spacy_entities")))
        else:
            spacy_pool.extend(current_spacy_pool)
    elif field == "place":
        add_values(semantic_pool, sem_idx, "place_norm")
        add_values(spatial_pool, spatial_idx, "place_norm")
        if current_spacy_pool is None:
            spacy_row = spacy_df.iloc[record_idx]
            spacy_pool.extend(json_to_list(spacy_row.get("spacy_places")))
            spacy_pool.extend(json_to_list(spacy_row.get("spacy_entities")))
        else:
            spacy_pool.extend(current_spacy_pool)
    elif field == "title":
        add_values(semantic_pool, sem_idx, "title_norm")
        add_values(spatial_pool, spatial_idx, "title_norm")
        if current_spacy_pool is None:
            spacy_row = spacy_df.iloc[record_idx]
            spacy_pool.extend(json_to_list(spacy_row.get("spacy_noun_chunks")))
        else:
            spacy_pool.extend(current_spacy_pool)
    elif field == "abstract":
        add_values(semantic_pool, sem_idx, "abstract_norm")
        add_values(spatial_pool, spatial_idx, "abstract_norm")
        if current_spacy_pool is None:
            spacy_row = spacy_df.iloc[record_idx]
            spacy_pool.extend(json_to_list(spacy_row.get("spacy_noun_chunks")))
            spacy_pool.extend(json_to_list(spacy_row.get("spacy_entities")))
        else:
            spacy_pool.extend(current_spacy_pool)
    else:
        raise ValueError(f"Unsupported field: {field}")

    return {
        "sem_idx": sem_idx,
        "spatial_idx": spatial_idx,
        "semantic_pool": semantic_pool,
        "spatial_pool": spatial_pool,
        "spacy_pool": spacy_pool,
    }


def _summarize_candidate_pool(values: list[str], top_n: int = 3) -> tuple[list[tuple[str, int]], Counter]:
    counter = Counter([normalize_text(v).lower() for v in values if normalize_text(v)])
    top = counter.most_common(top_n)
    return top, counter


def _make_abstract_summary(row: pd.Series, support: dict[str, Any]) -> str:
    semantic_pool = support.get("semantic_pool", [])
    if semantic_pool:
        sentence = max(semantic_pool, key=lambda s: len(str(s)))
        text = normalize_text(sentence) or ""
        if len(text.split()) > 40:
            text = " ".join(text.split()[:40])
        return text
    title = row.get("title_norm") or "the dataset"
    keywords = ", ".join((row.get("keywords_norm") or [])[:3])
    place = ", ".join((row.get("place_norm") or [])[:2])
    parts = [f"This dataset concerns {title.lower()}" if title else "This dataset is described"]
    if keywords:
        parts.append(f"It is associated with {keywords}.")
    if place:
        parts.append(f"Relevant geographic context includes {place}.")
    return " ".join(parts).strip()


def predict_for_mode(
    field: str,
    row: pd.Series,
    support: dict[str, Any],
    mode: str = "fused",
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    tau: float = DEFAULT_TAU,
) -> dict[str, Any]:
    semantic_pool = support["semantic_pool"] if mode in {"transformer", "fused"} else []
    spatial_pool = support["spatial_pool"] if mode == "fused" else []
    spacy_pool = support["spacy_pool"] if mode in {"spacy", "fused"} else []
    combined = semantic_pool + spatial_pool + spacy_pool
    if not combined:
        return {
            "predicted_value": None,
            "top3_candidates": json.dumps([], ensure_ascii=False),
            "confidence": 0.0,
            "action": "suggest",
            "sem_support_count": 0,
            "spatial_support_count": 0,
            "spacy_support_count": 0,
        }

    sem_counter = Counter([normalize_text(v).lower() for v in semantic_pool if normalize_text(v)])
    spatial_counter = Counter([normalize_text(v).lower() for v in spatial_pool if normalize_text(v)])
    spacy_counter = Counter([normalize_text(v).lower() for v in spacy_pool if normalize_text(v)])
    combined_counter = Counter([normalize_text(v).lower() for v in combined if normalize_text(v)])
    total = sum(combined_counter.values()) or 1

    scored = []
    for candidate, freq in combined_counter.items():
        sem_score = sem_counter.get(candidate, 0) / max(1, len(semantic_pool)) if semantic_pool else 0.0
        spatial_score = spatial_counter.get(candidate, 0) / max(1, len(spatial_pool)) if spatial_pool else 0.0
        freq_score = freq / total
        if candidate in spacy_counter:
            sem_score = max(sem_score, 0.5)
        score = _score_components(sem_score, spatial_score, freq_score, alpha, beta, gamma)
        scored.append((candidate, score, freq))

    scored.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
    top3 = scored[:3]
    predicted = top3[0][0] if top3 else None
    raw_confidence = float(top3[0][1]) if top3 else 0.0
    confidence_anchor = {"title": 0.12, "keywords": 0.18, "place": 0.18, "abstract": 0.12}.get(field, 0.20)
    confidence = _sigmoid(10.0 * (raw_confidence - confidence_anchor))

    if field == "abstract":
        predicted = _make_abstract_summary(row, support)
        confidence = min(confidence, 0.79)

    action = "auto-fill" if confidence >= tau and field != "abstract" else "suggest"
    top3_json = json.dumps([{"value": cand, "score": float(score), "freq": int(freq)} for cand, score, freq in top3], ensure_ascii=False)
    return {
        "predicted_value": predicted,
        "top3_candidates": top3_json,
        "confidence": confidence,
        "action": action,
        "sem_support_count": len(semantic_pool),
        "spatial_support_count": len(spatial_pool),
        "spacy_support_count": len(spacy_pool),
    }


def _build_prediction_for_field(field: str, row: pd.Series, support: dict[str, Any], alpha: float, beta: float, gamma: float, tau: float) -> dict[str, Any]:
    return predict_for_mode(field, row, support, mode="fused", alpha=alpha, beta=beta, gamma=gamma, tau=tau)


def aggregate_and_fill(
    preproc_path: str | Path | None = None,
    candidate_graph_path: str | Path | None = None,
    spacy_path: str | Path | None = None,
    out_long_path: str | Path | None = None,
    out_wide_path: str | Path | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    tau: float = DEFAULT_TAU,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preproc_path = Path(preproc_path or (PARSED_DIR / "canonical_preproc.parquet"))
    candidate_graph_path = Path(candidate_graph_path or (CANDIDATE_DIR / "candidate_graph.csv"))
    spacy_path = Path(spacy_path or (PARSED_DIR / SPACY_FILE))
    out_long_path = Path(out_long_path or (PARSED_DIR / LONG_SUGGESTIONS_FILE))
    out_wide_path = Path(out_wide_path or (PARSED_DIR / WIDE_SUGGESTIONS_FILE))

    df = load_dataframe(preproc_path)
    candidate_graph = load_dataframe(candidate_graph_path)
    spacy_df = load_dataframe(spacy_path)

    long_rows = []
    for i, row in df.iterrows():
        for field in TARGET_FIELDS:
            current = row.get(f"{field}_norm") if field in {"title", "abstract"} else row.get(f"{field}_norm")
            if field in {"keywords", "place"}:
                current_present = bool(row.get(f"has_{field}"))
            else:
                current_present = bool(normalize_text(current))
            support = _extract_neighbor_support(df, candidate_graph, spacy_df, i, field)
            pred = _build_prediction_for_field(field, row, support, alpha, beta, gamma, tau)
            long_rows.append(
                {
                    "filename": row["filename"],
                    "record_index": i,
                    "field": field,
                    "before_present": current_present,
                    "predicted_value": pred["predicted_value"],
                    "top3_candidates": pred["top3_candidates"],
                    "confidence": pred["confidence"],
                    "action": pred["action"],
                    "sem_support_count": pred["sem_support_count"],
                    "spatial_support_count": pred["spatial_support_count"],
                    "spacy_support_count": pred["spacy_support_count"],
                    "excluded_from_autofill": field in EXCLUDED_FIELDS,
                }
            )
    long_df = pd.DataFrame(long_rows)
    save_dataframe(long_df, out_long_path)

    wide_df = (
        long_df.pivot_table(index=["filename", "record_index"], columns="field", values=["predicted_value", "confidence", "action"], aggfunc="first")
        .reset_index()
    )
    wide_df.columns = ["_".join([str(c) for c in col if c != ""]).strip("_") if isinstance(col, tuple) else col for col in wide_df.columns]
    save_dataframe(wide_df, out_wide_path)
    return long_df, wide_df
