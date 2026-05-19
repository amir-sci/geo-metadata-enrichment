from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import EMBED_DIR, INDEX_DIR
from .utils import ensure_dir


class VectorIndex:
    def __init__(self, backend: str, model: Any, ids: np.ndarray):
        self.backend = backend
        self.model = model
        self.ids = ids

    def query(self, query_vectors: np.ndarray, k: int = 10):
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        if self.backend == "sklearn":
            distances, indices = self.model.kneighbors(query_vectors, n_neighbors=k)
            return indices, distances
        if self.backend == "faiss":
            D, I = self.model.search(query_vectors.astype(np.float32), k)
            return I, D
        if self.backend == "annoy":
            results = [self.model.get_nns_by_vector(vec.tolist(), k, include_distances=True) for vec in query_vectors]
            indices = np.array([r[0] for r in results], dtype=object)
            distances = np.array([r[1] for r in results], dtype=object)
            return indices, distances
        raise ValueError(f"Unsupported backend: {self.backend}")


def _build_faiss(embeddings: np.ndarray):
    try:
        import faiss  # type: ignore

        vecs = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(vecs)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return index, "faiss"
    except Exception:
        return None, None


def _build_annoy(embeddings: np.ndarray):
    try:
        from annoy import AnnoyIndex  # type: ignore

        d = embeddings.shape[1]
        index = AnnoyIndex(d, "angular")
        for i, vec in enumerate(embeddings):
            index.add_item(i, vec.tolist())
        index.build(50)
        return index, "annoy"
    except Exception:
        return None, None


def build_index(
    embeddings: np.ndarray | None = None,
    ids: np.ndarray | None = None,
    out_dir: str | Path | None = None,
    embeddings_path: str | Path | None = None,
    ids_path: str | Path | None = None,
) -> VectorIndex:
    out_dir = Path(out_dir or INDEX_DIR)
    ensure_dir(out_dir)
    if embeddings is None:
        embeddings = np.load(embeddings_path or (EMBED_DIR / "embeddings.npy"), allow_pickle=True)
    if ids is None:
        ids = np.load(ids_path or (EMBED_DIR / "ids.npy"), allow_pickle=True)

    faiss_index, backend = _build_faiss(embeddings)
    if faiss_index is not None:
        try:
            import faiss  # type: ignore
            vecs = embeddings.astype(np.float32).copy()
            faiss.normalize_L2(vecs)
            faiss.write_index(faiss_index, str(out_dir / "faiss.index"))
            (out_dir / "index_meta.json").write_text(json.dumps({"backend": "faiss", "n": len(ids)}, indent=2), encoding="utf-8")
            return VectorIndex("faiss", faiss_index, ids)
        except Exception:
            pass

    annoy_index, backend = _build_annoy(embeddings)
    if annoy_index is not None:
        annoy_index.save(str(out_dir / "annoy.index"))
        with open(out_dir / "annoy_ids.pkl", "wb") as f:
            pickle.dump(ids, f)
        (out_dir / "index_meta.json").write_text(json.dumps({"backend": "annoy", "n": len(ids)}, indent=2), encoding="utf-8")
        return VectorIndex("annoy", annoy_index, ids)

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    with open(out_dir / "sklearn_nn.pkl", "wb") as f:
        pickle.dump(nn, f)
    with open(out_dir / "sklearn_ids.pkl", "wb") as f:
        pickle.dump(ids, f)
    (out_dir / "index_meta.json").write_text(json.dumps({"backend": "sklearn", "n": len(ids)}, indent=2), encoding="utf-8")
    return VectorIndex("sklearn", nn, ids)


def load_index(out_dir: str | Path | None = None) -> VectorIndex:
    out_dir = Path(out_dir or INDEX_DIR)
    meta_path = out_dir / "index_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    backend = meta["backend"]
    if backend == "faiss":
        import faiss  # type: ignore
        index = faiss.read_index(str(out_dir / "faiss.index"))
        ids = np.load(EMBED_DIR / "ids.npy", allow_pickle=True)
        return VectorIndex("faiss", index, ids)
    if backend == "annoy":
        from annoy import AnnoyIndex  # type: ignore
        with open(out_dir / "annoy_ids.pkl", "rb") as f:
            ids = pickle.load(f)
        # dimension must be recovered from ids or embeddings file
        emb = np.load(EMBED_DIR / "embeddings.npy", allow_pickle=True)
        index = AnnoyIndex(emb.shape[1], "angular")
        index.load(str(out_dir / "annoy.index"))
        return VectorIndex("annoy", index, ids)
    with open(out_dir / "sklearn_nn.pkl", "rb") as f:
        nn = pickle.load(f)
    with open(out_dir / "sklearn_ids.pkl", "rb") as f:
        ids = pickle.load(f)
    return VectorIndex("sklearn", nn, ids)
