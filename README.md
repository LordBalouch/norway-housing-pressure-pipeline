# Norway Housing Pressure Tracker (1992–present)

A reproducible analytics pipeline and reviewer-ready deliverables to track Norwegian housing price dynamics over time:
trend (HPI level), growth regimes (YoY/QoQ), volatility regimes (QoQ variability by decade), and policy context (HPI vs Norges Bank policy rate).

## Business question
How have housing price growth and volatility evolved since 1992, and how do they relate to monetary policy context over time?

## Deliverables (open these first)
- Tableau dashboard exports (PNG): `tableau/exports/`
  - `tableau/exports/D1_executive_overview.png`
  - `tableau/exports/D2_growth.png`
  - `tableau/exports/D3_volatility.png`
  - `tableau/exports/D4_policy_context.png`
- Tableau workbook (Part 4): `tableau/norway_housing_pressure_part4.twb` / `tableau/norway_housing_pressure_part4.twbx`
- Excel workbook (Part 3): `excel/norway_housing_presure_part3.xlsx`
- SQL mart + validation + EDA scripts: `sql/`
- Run order / reproducibility: `docs/reproduce_project.md`
- Deliverables map: `docs/deliverables_index.md`
- Data dictionary: `docs/data_dictionary.md`
- Validation metadata: `docs/metadata_part1.json`, `docs/metadata_part2.json`
- Data sources notes: `docs/data_sources.md`

## How to reproduce
Follow the single run-order document:
- `docs/reproduce_project.md`

High-level sequence:
1. `python src/01_download_raw.py`
2. `python src/02_transform_clean.py --run all --tag <TAG>`
3. `python -u src/03_run_duckdb_sql.py --run-validation --run-eda`

## Repo structure
- `src/` — Python pipeline scripts
- `sql/` — DuckDB SQL mart, validation checks, EDA queries
- `excel/` — Excel deliverable (tracked)
- `tableau/` — Tableau workbook + dashboard exports (tracked)
- `docs/` — metadata, notes, data dictionary, navigation docs
- `data/raw/`, `data/processed/` — generated outputs (intentionally not committed)

## Limitations (current project state)
- CPI is annual and repeated quarterly; “real_*” metrics are proxies.
- CPI missing for 2025Q1–2025Q3.
- Mart currently includes one `boligtype_code` (no boligtype slicer).
- Tableau export workflow limitations are documented in `docs/tableau_part4_notes.md`.
