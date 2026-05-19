# Semantic Enrichment and Automatic Completion of Regional Geospatial Metadata Using NLP

This repository contains the code and data used in the paper on semantic enrichment and automatic completion of regional geospatial metadata using a comparative NLP pipeline.

## Overview
The pipeline parses XML metadata, normalizes fields into a canonical schema, extracts candidate values using spaCy and transformer-based semantic similarity, combines spatial and semantic evidence for ranking, and exports evaluation-ready outputs.

## Main components
- XML parsing and schema normalization
- Text preprocessing and canonical field mapping
- spaCy-based candidate extraction
- Transformer-based embedding and semantic retrieval
- Spatial neighbor retrieval with distance constraints
- Weighted ranking and confidence-based auto-fill
- Evaluation and report generation

## Repository structure
- `src/`: Python source code
- `config/`: pipeline configuration
- `data/`: source metadata and sample files
- `outputs/`: sample outputs and figures
- `README.md`: project documentation

## Expected outputs
-The pipeline generates reports such as:
	dataset summary tables
	coverage before/after tables
	auto-fill versus suggestion breakdowns
	comparative evaluation metrics
	ablation and sensitivity analysis results
	runtime benchmark reports
	annotation template files when needed

## Data availability
The repository includes the source metadata used in the study as well as the code needed to reproduce the main experiments. Some records may be excluded during XML validation if they are malformed, and the final analysis is based on the valid records only.

## Reproducibility notes
-To reproduce the reported results:
	Use the provided source metadata.
	Keep the default ranking parameters unless otherwise stated.
	Ensure that the same embedding model is selected.
	Run the pipeline and evaluation scripts in the documented order.

## Citation
If you use this repository, please cite the associated paper.

## Contact
For questions regarding the repository, please contact the corresponding author.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
