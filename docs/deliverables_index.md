# Deliverables Index — Norway Housing Pressure Tracker (1992–present)

## What to open first (2–3 minutes)
1. README: `README.md`
2. Tableau exports (quick visuals):
   - `tableau/exports/D1_executive_overview.png`
   - `tableau/exports/D2_growth.png`
   - `tableau/exports/D3_volatility.png`
   - `tableau/exports/D4_policy_context.png`
3. Excel workbook (Part 3):
   - `excel/norway_housing_presure_part3.xlsx`
4. How to reproduce the pipeline:
   - `docs/reproduce_project.md`
5. Definitions / schema:
   - `docs/data_dictionary.md`

## Part-by-part deliverables map

| Part | Purpose | Key files (committed) | Generated outputs (not committed) |
|---|---|---|---|
| Part 1 | Raw data download | `src/01_download_raw.py`, `docs/metadata_part1.json`, `docs/data_sources.md` | `data/raw/` |
| Part 2 | Transform + validation + documentation | `src/02_transform_clean.py`, `docs/metadata_part2.json`, `docs/data_dictionary.md` | `data/processed/` |
| Part 3 | DuckDB SQL mart + validation + EDA + Excel | `sql/01_build_mart.sql`, `sql/02_validation_checks.sql`, `sql/03_eda_queries.sql`, `src/03_run_duckdb_sql.py`, `excel/norway_housing_presure_part3.xlsx` | `data/processed/mart_housing_pressure_quarterly.csv` |
| Part 4 | Tableau dashboards (Mac) | `tableau/norway_housing_pressure_part4.twb`, `tableau/norway_housing_pressure_part4.twbx`, `tableau/exports/*.png`, `docs/tableau_part4_notes.md` | (Mart is reproducible; exports are committed) |

## Known limitations (current project state)
- CPI is annual and repeated quarterly; “real_*” metrics are proxies.
- CPI missing for 2025Q1–2025Q3.
- Mart currently includes one `boligtype_code` (so no boligtype slicer).
- Tableau export workflow limitations are documented in `docs/tableau_part4_notes.md`.
