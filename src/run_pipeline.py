from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .benchmark import save_benchmark_report, timer
from .candidate_gen import build_candidate_graph
from .config import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_K_SEM,
    DEFAULT_K_SPATIAL,
    DEFAULT_MAX_SPATIAL_DISTANCE_KM,
    DEFAULT_MODEL_NAME,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_TAU,
    EVALUATION_DIR,
    INPUT_XML_DIR,
    PARSED_DIR,
    REPORT_DIR,
    ANNOTATION_DIR,
)
from .embeddings import compute_embeddings
from .ann_index import build_index
from .parsing import parse_all_xml
from .preprocess import preprocess_df
from .ranking import aggregate_and_fill
from .report_tables import write_all_reports
from .spacy_pipeline import run_spacy
from .evaluate import evaluate_pipeline
from .annotation_agreement import prepare_annotation_template
from .utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Revised SDI-NLP pipeline")
    parser.add_argument("--input_dir", type=str, default=str(INPUT_XML_DIR), help="Directory containing XML metadata files")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Sentence-transformer model name")
    parser.add_argument("--k_sem", type=int, default=DEFAULT_K_SEM)
    parser.add_argument("--k_spatial", type=int, default=DEFAULT_K_SPATIAL)
    parser.add_argument("--max_distance_km", type=float, default=DEFAULT_MAX_SPATIAL_DISTANCE_KM)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--run_evaluation", action="store_true", help="Run masked recovery evaluation and export review-ready reports")
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--make_annotation_template", action="store_true", help="Create a manual annotation template in artifacts/annotations")
    parser.add_argument("--annotation_sample_size", type=int, default=120)
    parser.add_argument("--full_ablation", action="store_true", help="Include k-sweep candidates in ablation_results (slower)")
    args = parser.parse_args()

    bench = []
    for p in [REPORT_DIR, EVALUATION_DIR, ANNOTATION_DIR]:
        ensure_dir(p)

    with timer("parse_xml", rows=0, results=bench):
        valid_df, invalid_df = parse_all_xml(args.input_dir)

    with timer("preprocess", rows=len(valid_df), results=bench):
        preproc_df = preprocess_df()

    with timer("embeddings", rows=len(preproc_df), results=bench):
        embeddings, _, embed_meta = compute_embeddings(model_name=args.model_name)

    with timer("index_build", rows=len(preproc_df), results=bench):
        build_index(embeddings=embeddings)

    with timer("spacy_extraction", rows=len(preproc_df), results=bench):
        spacy_df = run_spacy()

    with timer("candidate_graph", rows=len(preproc_df), results=bench):
        candidate_graph = build_candidate_graph(k_sem=args.k_sem, k_spatial=args.k_spatial, max_distance_km=args.max_distance_km)

    with timer("ranking_fill", rows=len(preproc_df) * 4, results=bench):
        long_df, wide_df = aggregate_and_fill(alpha=args.alpha, beta=args.beta, gamma=args.gamma, tau=args.tau)

    with timer("report_tables", rows=len(preproc_df), results=bench):
        report_paths = write_all_reports(valid_df, long_df)

    if args.run_evaluation:
        with timer("masked_evaluation", rows=len(preproc_df), results=bench):
            eval_df = evaluate_pipeline(
                input_dir=args.input_dir,
                sample_size=args.sample_size,
                model_name=args.model_name,
                k_sem=args.k_sem,
                k_spatial=args.k_spatial,
                max_distance_km=args.max_distance_km,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                tau=args.tau,
            )
        print(f"Evaluation report: {EVALUATION_DIR}")

    if args.make_annotation_template:
        with timer("annotation_template", rows=min(args.annotation_sample_size, len(preproc_df)), results=bench):
            template = prepare_annotation_template(sample_size=args.annotation_sample_size)
        print(f"Annotation template: {ANNOTATION_DIR / 'annotation_template.csv'}")

    bench_df = pd.DataFrame(bench)
    save_path = save_benchmark_report(bench)
    print("Pipeline complete.")
    print(f"Valid records: {len(valid_df)} | Invalid records: {len(invalid_df)}")
    print(f"Artifacts saved under: {REPORT_DIR.parent}")
    print(f"Benchmark report: {save_path}")


if __name__ == "__main__":
    main()
