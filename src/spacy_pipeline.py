from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import PARSED_DIR, SPACY_FILE
from .utils import ensure_dir, load_dataframe, normalize_text, save_dataframe


@lru_cache(maxsize=1)
def _load_spacy_model():
    try:
        import spacy  # type: ignore

        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
    except Exception:
        return None


def _unique(seq):
    out = []
    seen = set()
    for x in seq:
        if not x:
            continue
        key = x.lower()
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def _extract_doc_features(doc):
    entities = []
    noun_chunks = []
    place_entities = []
    if doc is None:
        return entities, noun_chunks, place_entities
    try:
        entities = [normalize_text(ent.text) for ent in doc.ents if normalize_text(ent.text)]
        place_entities = [normalize_text(ent.text) for ent in doc.ents if getattr(ent, "label_", "") in {"GPE", "LOC", "FAC"} and normalize_text(ent.text)]
    except Exception:
        pass
    try:
        noun_chunks = [normalize_text(chunk.text) for chunk in doc.noun_chunks if normalize_text(chunk.text)]
    except Exception:
        noun_chunks = []
    return _unique(entities), _unique(noun_chunks), _unique(place_entities)


def extract_spacy_features(text: str | None) -> tuple[list[str], list[str], list[str]]:
    text = normalize_text(text)
    if not text:
        return [], [], []
    nlp = _load_spacy_model()
    if nlp is None:
        return [], [], []
    try:
        doc = nlp(text)
        return _extract_doc_features(doc)
    except Exception:
        return [], [], []


def run_spacy(input_path: str | Path | None = None, out_path: str | Path | None = None) -> pd.DataFrame:
    input_path = Path(input_path or (PARSED_DIR / "canonical_preproc.parquet"))
    out_path = Path(out_path or (PARSED_DIR / SPACY_FILE))
    ensure_dir(out_path.parent)
    df = load_dataframe(input_path)

    rows = []
    for _, r in df.iterrows():
        text = " . ".join([x for x in [r.get("title_norm"), r.get("abstract_norm"), " ".join(r.get("keywords_norm", [])), " ".join(r.get("place_norm", []))] if isinstance(x, str) and x.strip()])
        entities, noun_chunks, place_entities = extract_spacy_features(text)
        rows.append(
            {
                "filename": r["filename"],
                "spacy_entities": json.dumps(entities, ensure_ascii=False),
                "spacy_noun_chunks": json.dumps(noun_chunks, ensure_ascii=False),
                "spacy_places": json.dumps(place_entities, ensure_ascii=False),
                "spacy_entity_count": len(entities),
                "spacy_noun_chunk_count": len(noun_chunks),
            }
        )
    out = pd.DataFrame(rows)
    save_dataframe(out, out_path)
    return out


if __name__ == "__main__":
    run_spacy()
