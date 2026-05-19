from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_XML_DIR = PROJECT_ROOT / "Source"
WORK_DIR = PROJECT_ROOT / "artifacts"
RAW_DIR = WORK_DIR / "raw"
PARSED_DIR = WORK_DIR / "parsed"
EMBED_DIR = WORK_DIR / "embeddings"
INDEX_DIR = WORK_DIR / "indices"
CANDIDATE_DIR = WORK_DIR / "candidates"
REPORT_DIR = WORK_DIR / "reports"
EVALUATION_DIR = WORK_DIR / "evaluation"
ANNOTATION_DIR = WORK_DIR / "annotations"
FIGURE_DIR = WORK_DIR / "figures"
MODEL_DIR = WORK_DIR / "models"

DEFAULT_MODEL_NAME = "all-mpnet-base-v2"
DEFAULT_K_SEM = 10
DEFAULT_K_SPATIAL = 10
DEFAULT_MAX_SPATIAL_DISTANCE_KM = 250.0
DEFAULT_ALPHA = 0.60
DEFAULT_BETA = 0.25
DEFAULT_GAMMA = 0.15
DEFAULT_TAU = 0.80
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 300

EXCLUDED_AUTO_FIELDS = ("contact", "lineage", "provenance", "language", "pubdate", "date")
TARGET_FIELDS = ("title", "abstract", "keywords", "place")

PREFERRED_PLACE_TAGS = ("placekey", "placekt")
PREFERRED_KEYWORD_TAGS = ("themekey", "keyword")
PREFERRED_TITLE_TAGS = ("title",)
PREFERRED_ABSTRACT_TAGS = ("abstract", "purpose")

PARSING_ERRORS_FILE = "invalid_xml_files.csv"
SUMMARY_FILE = "dataset_summary.csv"
RAW_CANONICAL_FILE = "canonical_raw.csv"
PREPROC_FILE = "canonical_preproc.csv"
LONG_SUGGESTIONS_FILE = "filled_suggestions_long.csv"
WIDE_SUGGESTIONS_FILE = "filled_suggestions_wide.csv"
CANDIDATE_GRAPH_FILE = "candidate_graph.csv"
SPACY_FILE = "spacy_extractions.csv"
TABLE2_FILE = "table2_summary.csv"
TABLE3_FILE = "table3_coverage.csv"
TABLE4_FILE = "table4_examples.csv"
ABLATION_FILE = "ablation_results.csv"
BENCHMARK_FILE = "benchmark_report.csv"
KAPPA_FILE = "inter_annotator_agreement.csv"
COMPARISON_FILE = "comparison_metrics.csv"
ISOLATED_FILE = "isolated_records_analysis.csv"
CONFIDENCE_FILE = "confidence_statistics.csv"
ANNOTATION_TEMPLATE_FILE = "annotation_template.csv"
ANNOTATION_STATUS_FILE = "annotation_agreement_status.csv"
