## data/README.md

```md
# Data Description
This folder contains the source geospatial metadata used in the study, together with a small sample subset for quick testing and validation.

## Contents
- `source_metadata/` — XML metadata records used in the experiments
- `sample/` — a small subset of records for testing the pipeline structure
- `README.md` — this file

## Notes
The source XML files were parsed and normalized into a canonical schema before evaluation. During validation, malformed XML records were excluded from downstream analysis. Therefore, the final results reported in the paper are based only on valid metadata records.

## Recommended use
- Use `source_metadata/` for full reproduction of the study
- Use `sample/` for quick checks, syntax testing, and pipeline verification

## Important
Do not modify the original source metadata files if you want to reproduce the reported results exactly. Any changes to the XML files may alter the parsed counts, coverage values, and evaluation outputs.
