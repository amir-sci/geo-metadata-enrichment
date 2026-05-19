from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import DEFAULT_MODEL_NAME, EMBED_DIR, PARSED_DIR
from .utils import ensure_dir, load_dataframe, save_dataframe


class EmbeddingBackend:
    SENTENCE_TRANSFORMER = "sentence_transformer"
    TFIDF = "tfidf"


def build_corpus_text(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ["title_norm", "abstract_norm", "purpose_norm"]:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    for col in ["keywords_norm", "place_norm"]:
        val = row.get(col)
        if isinstance(val, list) and val:
            parts.append(" ".join(val))
    return " . ".join(parts)


def _try_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_name)
        return model
    except Exception:
        return None


def compute_embeddings(
    input_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    input_path = Path(input_path or (PARSED_DIR / "canonical_preproc.parquet"))
    out_dir = Path(out_dir or EMBED_DIR)
    ensure_dir(out_dir)

    df = load_dataframe(input_path)
    corpus = df.apply(build_corpus_text, axis=1).fillna("").tolist()

    st_model = _try_sentence_transformer(model_name)
    meta: dict[str, Any]
    if st_model is not None:
        embeddings = st_model.encode(corpus, show_progress_bar=True, convert_to_numpy=True, batch_size=32).astype(np.float32)
        backend = EmbeddingBackend.SENTENCE_TRANSFORMER
        meta = {"backend": backend, "model_name": model_name}
        np.save(out_dir / "embeddings.npy", embeddings)
        np.save(out_dir / "ids.npy", df["filename"].to_numpy())
        (out_dir / "embedding_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return embeddings, df, meta

    # Fallback for offline / test environments.
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=6000, min_df=1)
    X = vectorizer.fit_transform(corpus)
    X = normalize(X, norm="l2", axis=1)
    embeddings = X.toarray().astype(np.float32)
    backend = EmbeddingBackend.TFIDF
    meta = {"backend": backend, "model_name": model_name, "fallback": True, "vectorizer_type": "TfidfVectorizer"}
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "ids.npy", df["filename"].to_numpy())
    with open(out_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    (out_dir / "embedding_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return embeddings, df, meta


if __name__ == "__main__":
    compute_embeddings()
