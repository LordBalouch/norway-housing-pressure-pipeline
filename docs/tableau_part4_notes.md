# Tableau Part 4 Notes — Norway Housing Pressure Tracker (1992–present)

## Dataset used
- Mart CSV: `data/processed/mart_housing_pressure_quarterly.csv`
- Mart row count: 135 (quarters 1992Q1–2025Q3)
- Regenerated via:
  - `source .venv/bin/activate`
  - `python src/03_run_duckdb_sql.py --run-validation`

### Mart build + validation summary (evidence)
- Mart build summary showed:
  - row_count = 135
  - boligtype_count = 1
  - min_quarter_start = 1992-01-01
  - max_quarter_start = 2025-07-01
- Validation checks:
  - duplicate_key_rows = 0
  - missing_cpi_index = 3 rows (2025Q1–2025Q3)

## Refresh steps (after regenerating mart)
1. Activate venv:
   - `source .venv/bin/activate`
2. Rebuild mart + run validations:
   - `python src/03_run_duckdb_sql.py --run-validation`
3. Confirm export exists and row count is stable:
   - `wc -l data/processed/mart_housing_pressure_quarterly.csv` → 136 lines (header + 135 rows)
4. In Tableau:
   - Connect/refresh the data source pointing to `data/processed/mart_housing_pressure_quarterly.csv`
   - Confirm record count = 135

## Tableau calculated fields (Part 4)
- Quarter Label:
  - `DATETRUNC('quarter', [Quarter Start])`
- Decade:
  - `INT(DATEPART('year', [Quarter Start]) / 10) * 10`
- CPI Missing:
  - `ISNULL([Cpi Index])`
- Is Latest Quarter:
  - `[Quarter Start] = { FIXED : MAX([Quarter Start]) }`

### Percent scaling (important fix)
The mart growth fields were interpreted as “percentage points” in Tableau (e.g., 5.00 meaning 5%).
To display correctly as percentages, created display-safe fields:
- HPI YoY (decimal):
  - `[Hpi YoY Pct] / 100`
- HPI QoQ (decimal):
  - `[Hpi QoQ Pct] / 100`
- Policy Rate (decimal) (if stored as 5 meaning 5%):
  - `[Policy Rate End] / 100`

### Volatility (Decade LOD)
- Volatility QoQ (Decade LOD):
  - `{ FIXED [Decade] : STDEV([HPI QoQ (decimal)]) }`

## Worksheets built (core)
- W1 Trend – HPI Index (line: HPI index over time; optional SA line)
- W2 Growth – YoY & QoQ (dual-axis; QoQ includes 0 baseline; Decade filter used on dashboard)
- W3 Volatility – QoQ Std Dev by Decade (bar; Decade discrete dimension; LOD used; avoid SUM aggregation)
- W4 Policy Context – HPI vs Policy Rate (dual-axis line; tooltip includes growth metrics)
- (Optional) W5 CPI Missing – Diagnostics (table of quarters with null CPI)

## Dashboards delivered (required)
- D1 Executive Overview (KPI tiles + HPI trend + caveats)
- D2 Growth (YoY and QoQ lines + Decade filter + 0 baseline)
- D3 Volatility (volatility = std dev of QoQ within decade)
- D4 Policy Context (dual-axis HPI vs policy rate)

## Known caveats
- CPI is annual repeated quarterly → “real_*” measures are proxies.
- CPI missing for 2025Q1–2025Q3 → real proxies are null for those quarters.
- Only one boligtype_code category currently → boligtype slicer not meaningful; used decade/time segmentation instead.

## Issues encountered + fixes
- macOS terminal: `python` not found in shell; resolved by activating venv and using venv python.
- Tableau Public edition export: in-app export image was not available; published to Tableau Public and downloaded images from the web view.
- YoY/QoQ scaling: corrected by dividing by 100 and formatting as Percentage.
- Volatility worksheet initially showed one bar: corrected by converting Decade to a discrete dimension (not `SUM(Decade)`) and using AVG (not SUM) for the LOD measure.

## Repo exports (required)
- `tableau/exports/D1_executive_overview.png`
- `tableau/exports/D2_growth.png`
- `tableau/exports/D3_volatility.png`
- `tableau/exports/D4_policy_context.png`
