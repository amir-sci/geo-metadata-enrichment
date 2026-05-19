from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ann_index import build_index
from .candidate_gen import build_candidate_graph
from .config import (
    ANNOTATION_DIR,
    ANNOTATION_STATUS_FILE,
    ANNOTATION_TEMPLATE_FILE,
    CANDIDATE_DIR,
    COMPARISON_FILE,
    CONFIDENCE_FILE,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_K_SEM,
    DEFAULT_K_SPATIAL,
    DEFAULT_MAX_SPATIAL_DISTANCE_KM,
    DEFAULT_MODEL_NAME,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_TAU,
    EVALUATION_DIR,
    ISOLATED_FILE,
    KAPPA_FILE,
    PARSED_DIR,
    TARGET_FIELDS,
    ABLATION_FILE,
)
from .embeddings import compute_embeddings
from .parsing import parse_all_xml
from .preprocess import preprocess_df
from .ranking import aggregate_and_fill, _extract_neighbor_support, predict_for_mode
from .spacy_pipeline import extract_spacy_features, run_spacy
from .utils import bootstrap_ci, ensure_dir, json_to_list, load_dataframe, normalize_text, save_dataframe


MODES = ("spacy", "transformer", "fused")


def _token_f1(true_text: str | None, pred_text: str | None) -> float:
    true_tokens = set((normalize_text(true_text) or "").lower().split())
    pred_tokens = set((normalize_text(pred_text) or "").lower().split())
    if not true_tokens and not pred_tokens:
        return 1.0
    if not true_tokens or not pred_tokens:
        return 0.0
    tp = len(true_tokens & pred_tokens)
    prec = tp / len(pred_tokens)
    rec = tp / len(true_tokens)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _list_metrics(true_items: list[str], pred_items: list[str], k: int = 3) -> dict[str, float]:
    true_norm = [normalize_text(x).lower() for x in true_items if normalize_text(x)]
    pred_norm = [normalize_text(x).lower() for x in pred_items if normalize_text(x)]
    true_set = set(true_norm)
    pred_set = set(pred_norm[:k]) if pred_norm else set()
    if not true_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not true_set or not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _sample_records(df: pd.DataFrame, n: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_RANDOM_SEED) -> pd.DataFrame:
    n = min(n, len(df))
    strata = df["dataset_group"].astype(str) + "|" + df["has_bbox"].astype(str)
    df = df.copy()
    df["_strata"] = strata
    if len(df["_strata"].unique()) <= 1 or n >= len(df):
        sample = df.sample(n=n, random_state=seed)
    else:
        parts = []
        total = len(df)
        group_sizes = df.groupby("_strata").size().sort_values(ascending=False)
        for i, (grp, size) in enumerate(group_sizes.items()):
            quota = max(1, int(round(n * size / total)))
            grp_df = df[df["_strata"] == grp]
            quota = min(quota, len(grp_df))
            parts.append(grp_df.sample(n=quota, random_state=seed + i))
        sample = pd.concat(parts, ignore_index=False).drop_duplicates(subset=["filename"]).head(n)
        if len(sample) < n:
            extra = df[~df["filename"].isin(sample["filename"])].sample(n=n - len(sample), random_state=seed)
            sample = pd.concat([sample, extra], ignore_index=False)
        sample = sample.sample(frac=1.0, random_state=seed)
    sample = sample.copy()
    sample["record_index"] = sample.index
    sample = sample.reset_index(drop=True)
    return sample


def _ensure_artifacts(input_dir: str | Path | None = None, model_name: str = DEFAULT_MODEL_NAME):
    raw_path = PARSED_DIR / "canonical_preproc.parquet"
    if not raw_path.exists() and not raw_path.with_suffix(".csv").exists():
        parse_all_xml(input_dir=input_dir)
        preprocess_df()
    preproc_df = load_dataframe(raw_path)
    emb_path = Path(EVALUATION_DIR.parent / "embeddings" / "embeddings.npy")
    if not emb_path.exists():
        embeddings, _, _ = compute_embeddings(model_name=model_name)
    else:
        embeddings = np.load(emb_path, allow_pickle=True)
    if not (Path(EVALUATION_DIR.parent) / "indices" / "index_meta.json").exists():
        build_index(embeddings=embeddings)
    spacy_path = PARSED_DIR / "spacy_extractions.csv"
    if not spacy_path.exists():
        run_spacy()
    candidate_path = CANDIDATE_DIR / "candidate_graph.csv"
    if not candidate_path.exists():
        build_candidate_graph(k_sem=DEFAULT_K_SEM, k_spatial=DEFAULT_K_SPATIAL, max_distance_km=DEFAULT_MAX_SPATIAL_DISTANCE_KM)
    long_path = PARSED_DIR / "filled_suggestions_long.csv"
    if not long_path.exists():
        aggregate_and_fill(alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, gamma=DEFAULT_GAMMA, tau=DEFAULT_TAU)
    return preproc_df


def _masked_text_for_field(row: pd.Series, field: str) -> str:
    parts: list[str] = []
    if field != "title" and normalize_text(row.get("title_norm")):
        parts.append(row.get("title_norm"))
    if field != "abstract" and normalize_text(row.get("abstract_norm")):
        parts.append(row.get("abstract_norm"))
    if field != "keywords":
        kws = row.get("keywords_norm") or []
        if isinstance(kws, list) and kws:
            parts.append(" ".join(kws))
    if field != "place":
        places = row.get("place_norm") or []
        if isinstance(places, list) and places:
            parts.append(" ".join(places))
    purpose = row.get("purpose_norm")
    if normalize_text(purpose):
        parts.append(purpose)
    return " . ".join([p for p in parts if normalize_text(p)])


def _row_true_value(row: pd.Series, field: str):
    if field in {"keywords", "place"}:
        return row.get(f"{field}_norm") if isinstance(row.get(f"{field}_norm"), list) else []
    return row.get(f"{field}_norm")


def _evaluate_detail_rows(preproc_df: pd.DataFrame, candidate_graph: pd.DataFrame, sample_df: pd.DataFrame, *, mode: str, alpha: float, beta: float, gamma: float, tau: float) -> pd.DataFrame:
    rows = []
    for _, sampled in sample_df.iterrows():
        record_index = int(sampled["record_index"])
        base_row = preproc_df.iloc[record_index].copy()
        is_isolated = bool(candidate_graph.iloc[record_index].get("is_spatially_isolated", False))
        for field in TARGET_FIELDS:
            masked_row = base_row.copy()
            if field in {"keywords", "place"}:
                masked_row[f"{field}_norm"] = []
                masked_row[f"has_{field}"] = False
            else:
                masked_row[f"{field}_norm"] = None
            masked_text = _masked_text_for_field(masked_row, field)
            entities, noun_chunks, places = extract_spacy_features(masked_text)
            current_spacy_pool = noun_chunks + entities if field != "place" else places + entities
            support = _extract_neighbor_support(
                preproc_df,
                candidate_graph,
                pd.DataFrame([{"spacy_entities": json.dumps(entities, ensure_ascii=False), "spacy_noun_chunks": json.dumps(noun_chunks, ensure_ascii=False), "spacy_places": json.dumps(places, ensure_ascii=False)}]),
                record_index,
                field,
                current_spacy_pool=current_spacy_pool,
            )
            pred = predict_for_mode(field, masked_row, support, mode=mode, alpha=alpha, beta=beta, gamma=gamma, tau=tau)
            true_value = _row_true_value(base_row, field)
            if field in {"keywords", "place"}:
                try:
                    pred_items = [x["value"] for x in json.loads(pred["top3_candidates"])]
                except Exception:
                    pred_items = []
                m = _list_metrics(true_value if isinstance(true_value, list) else [], pred_items, k=3)
            else:
                m = {"precision": np.nan, "recall": np.nan, "f1": _token_f1(true_value, pred["predicted_value"])}
            rows.append(
                {
                    "filename": base_row["filename"],
                    "record_index": record_index,
                    "dataset_group": base_row.get("dataset_group"),
                    "field": field,
                    "mode": mode,
                    "is_spatially_isolated": is_isolated,
                    "before_present": bool(base_row.get(f"has_{field}")) if field in {"keywords", "place"} else bool(normalize_text(base_row.get(f"{field}_norm"))),
                    "predicted_value": pred["predicted_value"],
                    "confidence": float(pred["confidence"]),
                    "action": pred["action"],
                    "sem_support_count": int(pred["sem_support_count"]),
                    "spatial_support_count": int(pred["spatial_support_count"]),
                    "spacy_support_count": int(pred["spacy_support_count"]),
                    **m,
                }
            )
    return pd.DataFrame(rows)


def _summarize_metrics(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in TARGET_FIELDS:
        for mode in MODES:
            sub = detail_df[(detail_df["field"] == field) & (detail_df["mode"] == mode)].copy()
            if sub.empty:
                continue
            for metric in ["precision", "recall", "f1"]:
                vals = sub[metric].dropna().tolist()
                if not vals:
                    continue
                lo, hi = bootstrap_ci(vals)
                rows.append(
                    {
                        "field": field,
                        "mode": mode,
                        "metric": metric,
                        "mean": float(np.mean(vals)),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n": len(vals),
                    }
                )
            rows.append(
                {
                    "field": field,
                    "mode": mode,
                    "metric": "auto_fill_rate",
                    "mean": float((sub["action"] == "auto-fill").mean()),
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "n": int(len(sub)),
                }
            )
            rows.append(
                {
                    "field": field,
                    "mode": mode,
                    "metric": "confidence_mean",
                    "mean": float(sub["confidence"].mean()),
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "n": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def _confidence_stats(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (field, mode, action), sub in detail_df.groupby(["field", "mode", "action"]):
        rows.append(
            {
                "field": field,
                "mode": mode,
                "action": action,
                "n": int(len(sub)),
                "mean": float(sub["confidence"].mean()),
                "median": float(sub["confidence"].median()),
                "std": float(sub["confidence"].std(ddof=0) if len(sub) > 1 else 0.0),
                "min": float(sub["confidence"].min()),
                "max": float(sub["confidence"].max()),
            }
        )
    return pd.DataFrame(rows)


def _isolated_analysis(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (field, mode, isolated), sub in detail_df.groupby(["field", "mode", "is_spatially_isolated"]):
        rows.append(
            {
                "field": field,
                "mode": mode,
                "is_spatially_isolated": bool(isolated),
                "n": int(len(sub)),
                "mean_precision": float(sub["precision"].dropna().mean()) if sub["precision"].notna().any() else np.nan,
                "mean_recall": float(sub["recall"].dropna().mean()) if sub["recall"].notna().any() else np.nan,
                "mean_f1": float(sub["f1"].dropna().mean()) if sub["f1"].notna().any() else np.nan,
                "mean_confidence": float(sub["confidence"].mean()),
                "auto_fill_rate": float((sub["action"] == "auto-fill").mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_ablation_configs(base_k_sem: int, base_k_spatial: int, base_alpha: float, base_beta: float, base_gamma: float, base_tau: float, include_k_sweep: bool = False):
    configs = [
        {"config": "baseline", "k_sem": base_k_sem, "k_spatial": base_k_spatial, "alpha": base_alpha, "beta": base_beta, "gamma": base_gamma, "tau": base_tau},
        {"config": "semantic_heavy", "k_sem": base_k_sem, "k_spatial": base_k_spatial, "alpha": 0.70, "beta": 0.15, "gamma": 0.15, "tau": base_tau},
        {"config": "spatial_heavy", "k_sem": base_k_sem, "k_spatial": base_k_spatial, "alpha": 0.50, "beta": 0.35, "gamma": 0.15, "tau": base_tau},
        {"config": "low_tau", "k_sem": base_k_sem, "k_spatial": base_k_spatial, "alpha": base_alpha, "beta": base_beta, "gamma": base_gamma, "tau": 0.70},
        {"config": "high_tau", "k_sem": base_k_sem, "k_spatial": base_k_spatial, "alpha": base_alpha, "beta": base_beta, "gamma": base_gamma, "tau": 0.90},
    ]
    if include_k_sweep:
        configs.extend([
            {"config": "small_k", "k_sem": 5, "k_spatial": 5, "alpha": base_alpha, "beta": base_beta, "gamma": base_gamma, "tau": base_tau},
            {"config": "large_k", "k_sem": 15, "k_spatial": 15, "alpha": base_alpha, "beta": base_beta, "gamma": base_gamma, "tau": base_tau},
        ])
    return configs


def _run_ablation_sweep(preproc_df: pd.DataFrame, sample_df: pd.DataFrame, model_name: str, base_candidate_graph: pd.DataFrame, base_spacy_df: pd.DataFrame, base_k_sem: int, base_k_spatial: int, max_distance_km: float, base_alpha: float, base_beta: float, base_gamma: float, base_tau: float, include_k_sweep: bool = False) -> pd.DataFrame:
    rows = []
    for cfg in _build_ablation_configs(base_k_sem, base_k_spatial, base_alpha, base_beta, base_gamma, base_tau, include_k_sweep=include_k_sweep):
        if cfg["k_sem"] == base_k_sem and cfg["k_spatial"] == base_k_spatial:
            candidate_graph = base_candidate_graph
        else:
            candidate_graph = build_candidate_graph(k_sem=cfg["k_sem"], k_spatial=cfg["k_spatial"], max_distance_km=max_distance_km, out_path=EVALUATION_DIR / f'candidate_graph_{cfg["config"]}.csv')
        detail_frames = []
        for mode in MODES:
            detail_frames.append(_evaluate_detail_rows(preproc_df, candidate_graph, sample_df, mode=mode, alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"], tau=cfg["tau"]))
        detail_df = pd.concat(detail_frames, ignore_index=True)
        summary_df = _summarize_metrics(detail_df)
        for _, r in summary_df.iterrows():
            rows.append(
                {
                    "config": cfg["config"],
                    "k_sem": cfg["k_sem"],
                    "k_spatial": cfg["k_spatial"],
                    "alpha": cfg["alpha"],
                    "beta": cfg["beta"],
                    "gamma": cfg["gamma"],
                    "tau": cfg["tau"],
                    **r.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def evaluate_pipeline(
    input_dir: str | Path | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
    model_name: str = DEFAULT_MODEL_NAME,
    k_sem: int = DEFAULT_K_SEM,
    k_spatial: int = DEFAULT_K_SPATIAL,
    max_distance_km: float = DEFAULT_MAX_SPATIAL_DISTANCE_KM,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    tau: float = DEFAULT_TAU,
) -> pd.DataFrame:
    ensure_dir(EVALUATION_DIR)
    preproc_df = _ensure_artifacts(input_dir=input_dir, model_name=model_name)
    candidate_graph = load_dataframe(CANDIDATE_DIR / "candidate_graph.csv")
    spacy_df = load_dataframe(PARSED_DIR / "spacy_extractions.csv")
    sample_df = _sample_records(preproc_df, n=sample_size, seed=seed)

    detail_frames = []
    for mode in MODES:
        detail_frames.append(_evaluate_detail_rows(preproc_df, candidate_graph, sample_df, mode=mode, alpha=alpha, beta=beta, gamma=gamma, tau=tau))
    detail_df = pd.concat(detail_frames, ignore_index=True)
    save_dataframe(detail_df, EVALUATION_DIR / "detailed_predictions.csv")

    comparison_df = _summarize_metrics(detail_df)
    save_dataframe(comparison_df, EVALUATION_DIR / COMPARISON_FILE)

    confidence_df = _confidence_stats(detail_df)
    save_dataframe(confidence_df, EVALUATION_DIR / CONFIDENCE_FILE)

    isolated_df = _isolated_analysis(detail_df)
    save_dataframe(isolated_df, EVALUATION_DIR / ISOLATED_FILE)

    ablation_df = _run_ablation_sweep(preproc_df, sample_df, model_name, candidate_graph, spacy_df, k_sem, k_spatial, max_distance_km, alpha, beta, gamma, tau, include_k_sweep=False)
    save_dataframe(ablation_df, EVALUATION_DIR / ABLATION_FILE)

    return comparison_df


if __name__ == "__main__":
    print(evaluate_pipeline().head())
