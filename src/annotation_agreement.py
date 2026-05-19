from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .config import ANNOTATION_DIR, ANNOTATION_TEMPLATE_FILE, ANNOTATION_STATUS_FILE, EVALUATION_DIR, KAPPA_FILE, PARSED_DIR, TARGET_FIELDS
from .preprocess import preprocess_df
from .utils import ensure_dir, load_dataframe, normalize_text, save_dataframe


LABEL_VALUES = ("correct", "partial", "incorrect", "unknown")


def _stratified_sample(df: pd.DataFrame, n: int = 120, seed: int = 42) -> pd.DataFrame:
    n = min(n, len(df))
    strata = df["dataset_group"].astype(str) + "|" + df["has_bbox"].astype(str)
    work = df.copy()
    work["_strata"] = strata
    if len(work["_strata"].unique()) <= 1 or n >= len(work):
        return work.sample(n=n, random_state=seed)
    parts = []
    total = len(work)
    group_sizes = work.groupby("_strata").size().sort_values(ascending=False)
    for i, (grp, size) in enumerate(group_sizes.items()):
        quota = max(1, int(round(n * size / total)))
        grp_df = work[work["_strata"] == grp]
        quota = min(quota, len(grp_df))
        parts.append(grp_df.sample(n=quota, random_state=seed + i))
    sample = pd.concat(parts, ignore_index=False).drop_duplicates(subset=["filename"]).head(n)
    if len(sample) < n:
        extra = work[~work["filename"].isin(sample["filename"])].sample(n=n - len(sample), random_state=seed)
        sample = pd.concat([sample, extra], ignore_index=False)
    return sample.sample(frac=1.0, random_state=seed)


def prepare_annotation_template(
    input_path: str | Path | None = None,
    out_path: str | Path | None = None,
    sample_size: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    input_path = Path(input_path or (PARSED_DIR / "canonical_preproc.parquet"))
    out_path = Path(out_path or (ANNOTATION_DIR / ANNOTATION_TEMPLATE_FILE))
    ensure_dir(out_path.parent)
    df = load_dataframe(input_path)
    sample = _stratified_sample(df, n=sample_size, seed=seed)

    rows = []
    for _, row in sample.iterrows():
        for field in TARGET_FIELDS:
            reference = row.get(f"{field}_norm") if field in {"title", "abstract"} else row.get(f"{field}_norm")
            if field in {"keywords", "place"}:
                reference = "; ".join(reference or []) if isinstance(reference, list) else ""
            rows.append(
                {
                    "filename": row["filename"],
                    "field": field,
                    "reference_value": reference if reference is not None else "",
                    "annotator_1_label": "",
                    "annotator_2_label": "",
                    "annotator_1_notes": "",
                    "annotator_2_notes": "",
                    "adjudicated_label": "",
                }
            )
    out = pd.DataFrame(rows)
    save_dataframe(out, out_path)

    status = pd.DataFrame([
        {
            "status": "template_created",
            "template_path": str(out_path),
            "label_scheme": ", ".join(LABEL_VALUES),
            "note": "Fill annotator_1_label and annotator_2_label, then run compute_kappa on the two completed CSVs.",
        }
    ])
    save_dataframe(status, EVALUATION_DIR / ANNOTATION_STATUS_FILE)
    return out


def compute_kappa(
    annotator1_path: str | Path,
    annotator2_path: str | Path,
    id_cols: list[str] | None = None,
    label_cols: list[str] | None = None,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    a1 = pd.read_csv(annotator1_path)
    a2 = pd.read_csv(annotator2_path)
    id_cols = id_cols or ["filename", "field"]
    label_cols = label_cols or [c for c in a1.columns if c not in set(id_cols) | {"reference_value", "annotator_1_notes", "annotator_2_notes", "adjudicated_label"}]
    merged = a1.merge(a2, on=id_cols, suffixes=("_a1", "_a2"))
    rows = []
    for col in label_cols:
        if f"{col}_a1" not in merged.columns or f"{col}_a2" not in merged.columns:
            continue
        s1 = merged[f"{col}_a1"].astype(str).str.strip().str.lower()
        s2 = merged[f"{col}_a2"].astype(str).str.strip().str.lower()
        valid = ~(s1.isin({"", "nan", "none"}) | s2.isin({"", "nan", "none"}))
        s1 = s1[valid]
        s2 = s2[valid]
        if len(s1) == 0:
            continue
        kappa = cohen_kappa_score(s1, s2)
        rows.append({"field": col, "cohen_kappa": float(kappa), "n": int(len(s1))})
    out = pd.DataFrame(rows)
    ensure_dir(EVALUATION_DIR)
    out_file = Path(out_path or (EVALUATION_DIR / KAPPA_FILE))
    save_dataframe(out, out_file)
    return out


def main():
    parser = argparse.ArgumentParser(description="Annotation agreement utilities")
    parser.add_argument("--template", action="store_true", help="Create an annotation template from the parsed corpus")
    parser.add_argument("--input_path", type=str, default=str(PARSED_DIR / "canonical_preproc.parquet"))
    parser.add_argument("--out_path", type=str, default=str(ANNOTATION_DIR / ANNOTATION_TEMPLATE_FILE))
    parser.add_argument("--sample_size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotator1", type=str, default=None)
    parser.add_argument("--annotator2", type=str, default=None)
    args = parser.parse_args()

    if args.template:
        out = prepare_annotation_template(args.input_path, args.out_path, sample_size=args.sample_size, seed=args.seed)
        print(out.head())
        return
    if args.annotator1 and args.annotator2:
        out = compute_kappa(args.annotator1, args.annotator2)
        print(out)
        return
    raise SystemExit("Provide --template or both --annotator1 and --annotator2.")


if __name__ == "__main__":
    main()
