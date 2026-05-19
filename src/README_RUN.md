# Revised SDI-NLP pipeline

## Inputs
- Default XML input directory: `Source/`

## Main outputs
The pipeline writes artifacts under `artifacts/`:
- `artifacts/parsed/`
  - `canonical_raw.csv` / `canonical_raw.parquet`
  - `canonical_preproc.csv` / `canonical_preproc.parquet`
  - `invalid_xml_files.csv`
  - `dataset_summary.csv`
  - `spacy_extractions.csv`
  - `candidate_graph.csv`
  - `filled_suggestions_long.csv`
  - `filled_suggestions_wide.csv`
- `artifacts/reports/`
  - `table2_summary.csv`
  - `table3_coverage.csv`
  - `table4_examples.csv`
- `artifacts/evaluation/`
  - `detailed_predictions.csv`
  - `comparison_metrics.csv`
  - `confidence_statistics.csv`
  - `isolated_records_analysis.csv`
  - `ablation_results.csv`
  - `benchmark_report.csv`
  - `annotation_agreement_status.csv`
- `artifacts/annotations/`
  - `annotation_template.csv` (created when `--make_annotation_template` is used)

## Run
```bash
python -m src.run_pipeline --input_dir ../Source --run_evaluation --make_annotation_template
```

If `sentence-transformers`, `faiss`, or `annoy` are not available, the code falls back to a sklearn-based implementation automatically.
