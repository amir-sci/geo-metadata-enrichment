from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ann_index import build_index, load_index
from .config import CANDIDATE_DIR, DEFAULT_K_SEM, DEFAULT_K_SPATIAL, DEFAULT_MAX_SPATIAL_DISTANCE_KM, EMBED_DIR, PARSED_DIR
from .utils import ensure_dir, json_to_list, load_dataframe, save_dataframe


def _haversine_km(lat1, lon1, lat2, lon2):
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return float("inf")
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(a)))


def get_semantic_candidates(index, query_vector: np.ndarray, k: int = DEFAULT_K_SEM) -> tuple[list[int], list[float]]:
    indices, distances = index.query(query_vector.reshape(1, -1), k=k + 1)
    idxs = indices[0].tolist() if isinstance(indices, np.ndarray) else list(indices[0])
    dists = distances[0].tolist() if isinstance(distances, np.ndarray) else list(distances[0])
    return idxs, dists


def compute_spatial_neighbors(df: pd.DataFrame, idx: int, k: int = DEFAULT_K_SPATIAL, max_distance_km: float | None = DEFAULT_MAX_SPATIAL_DISTANCE_KM) -> tuple[list[int], list[float]]:
    row = df.iloc[idx]
    if pd.isna(row.get("centroid_lat")) or pd.isna(row.get("centroid_lon")):
        return [], []
    base_lat = float(row["centroid_lat"])
    base_lon = float(row["centroid_lon"])
    candidates = []
    for j, other in df.iterrows():
        if j == idx:
            continue
        if pd.isna(other.get("centroid_lat")) or pd.isna(other.get("centroid_lon")):
            continue
        dist_km = _haversine_km(base_lat, base_lon, float(other["centroid_lat"]), float(other["centroid_lon"]))
        if max_distance_km is None or dist_km <= max_distance_km:
            candidates.append((j, dist_km))
    candidates.sort(key=lambda x: x[1])
    top = candidates[:k]
    return [i for i, _ in top], [d for _, d in top]


def build_candidate_graph(
    preproc_path: str | Path | None = None,
    embeddings_path: str | Path | None = None,
    out_path: str | Path | None = None,
    k_sem: int = DEFAULT_K_SEM,
    k_spatial: int = DEFAULT_K_SPATIAL,
    max_distance_km: float | None = DEFAULT_MAX_SPATIAL_DISTANCE_KM,
) -> pd.DataFrame:
    preproc_path = Path(preproc_path or (PARSED_DIR / "canonical_preproc.parquet"))
    out_path = Path(out_path or (CANDIDATE_DIR / "candidate_graph.csv"))
    ensure_dir(out_path.parent)
    df = load_dataframe(preproc_path)

    embeddings = np.load(embeddings_path or (EMBED_DIR / "embeddings.npy"), allow_pickle=True)
    index = load_index()

    # Batch semantic retrieval for speed.
    sem_indices, sem_distances = index.query(embeddings, k=k_sem + 1)

    rows = []
    for i, row in df.iterrows():
        idxs = sem_indices[i].tolist() if hasattr(sem_indices[i], "tolist") else list(sem_indices[i])
        dists = sem_distances[i].tolist() if hasattr(sem_distances[i], "tolist") else list(sem_distances[i])
        sem_pairs = [(int(j), float(d)) for j, d in zip(idxs, dists) if int(j) != i][:k_sem]
        spatial_idx, spatial_dist = compute_spatial_neighbors(df, i, k=k_spatial, max_distance_km=max_distance_km)
        rows.append(
            {
                "filename": row["filename"],
                "record_index": i,
                "semantic_neighbor_indices": json.dumps([j for j, _ in sem_pairs], ensure_ascii=False),
                "semantic_neighbor_distances": json.dumps([d for _, d in sem_pairs], ensure_ascii=False),
                "spatial_neighbor_indices": json.dumps(spatial_idx, ensure_ascii=False),
                "spatial_neighbor_distances_km": json.dumps(spatial_dist, ensure_ascii=False),
                "semantic_neighbor_count": len(sem_pairs),
                "spatial_neighbor_count": len(spatial_idx),
                "is_spatially_isolated": len(spatial_idx) == 0,
            }
        )
    out = pd.DataFrame(rows)
    save_dataframe(out, out_path)
    return out


if __name__ == "__main__":
    build_candidate_graph()
