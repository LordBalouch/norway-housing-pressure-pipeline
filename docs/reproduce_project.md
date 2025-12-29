# Reproduce the Project (Run Order)

This repository builds the Norway Housing Pressure Tracker (1992–present) end-to-end:

Raw download → Transform/Clean/Validate → DuckDB SQL Mart → Excel + Tableau deliverables

## 0) Environment setup (Mac)

From the repository root:

python -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  

## 1) Part 1 — Download raw data

python src/01_download_raw.py

Outputs (intentionally NOT committed):
- data/raw/ (timestamped extracts)

## 2) Part 2 — Transform, clean, and validate

python src/02_transform_clean.py --run all --tag <TAG>

Outputs (intentionally NOT committed):
- data/processed/ (cleaned datasets + derived tables)

Current project caveats:
- CPI is annual and repeated quarterly in the mart (proxy approach for “real_*” metrics).
- CPI missing for 2025Q1–2025Q3 in the current build.

## 3) Part 3 — Build DuckDB mart + run validation + EDA

python -u src/03_run_duckdb_sql.py --run-validation --run-eda

Key output (intentionally NOT committed):
- data/processed/mart_housing_pressure_quarterly.csv

Expected validation profile (current project state):
- row_count ≈ 135 (1992Q1–2025Q3)
- duplicate keys = 0
- min quarter_start = 1992-01-01
- max quarter_start = 2025-07-01 (2025Q3)

## 4) Deliverables (committed)

Excel (Part 3)
- excel/norway_housing_presure_part3.xlsx

Tableau (Part 4)
- Workbook files:
  - tableau/norway_housing_pressure_part4.twb
  - tableau/norway_housing_pressure_part4.twbx
- PNG exports:
  - tableau/exports/D1_executive_overview.png
  - tableau/exports/D2_growth.png
  - tableau/exports/D3_volatility.png
  - tableau/exports/D4_policy_context.png
- Notes:
  - docs/tableau_part4_notes.md

## Troubleshooting (Mac)

- If a script fails with missing packages: re-activate venv and re-run pip install -r requirements.txt
- If relative paths fail: confirm you are in repo root (pwd) before running commands
- If data/raw or data/processed are missing: run Part 1 then Part 2
