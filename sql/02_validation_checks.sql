-- 02_validation_checks.sql
-- Purpose: Data-quality validation checks for the quarterly mart.
-- Assumes mart_housing_pressure_quarterly exists in the connected DuckDB database.

-- 1) Row counts + key uniqueness (expect duplicate_key_rows = 0)
SELECT
  COUNT(*) AS row_count,
  COUNT(DISTINCT boligtype_code) AS boligtype_count,
  COUNT(DISTINCT quarter) AS quarter_count,
  COUNT(DISTINCT (CAST(boligtype_code AS VARCHAR) || '|' || quarter)) AS distinct_key_count,
  COUNT(*) - COUNT(DISTINCT (CAST(boligtype_code AS VARCHAR) || '|' || quarter)) AS duplicate_key_rows
FROM mart_housing_pressure_quarterly;

-- 2) Duplicate keys detail (expect 0 rows)
SELECT boligtype_code, quarter, COUNT(*) AS n
FROM mart_housing_pressure_quarterly
GROUP BY 1,2
HAVING COUNT(*) > 1
ORDER BY n DESC, boligtype_code, quarter;

-- 3) Missingness checks for key fields (expect 0 for all below EXCEPT CPI may be missing in latest quarters)
SELECT
  SUM(CASE WHEN boligtype_code IS NULL THEN 1 ELSE 0 END) AS missing_boligtype_code,
  SUM(CASE WHEN quarter IS NULL OR TRIM(quarter) = '' THEN 1 ELSE 0 END) AS missing_quarter,
  SUM(CASE WHEN quarter_start IS NULL THEN 1 ELSE 0 END) AS missing_quarter_start,
  SUM(CASE WHEN hpi_index IS NULL THEN 1 ELSE 0 END) AS missing_hpi_index,
  SUM(CASE WHEN cpi_index IS NULL THEN 1 ELSE 0 END) AS missing_cpi_index
FROM mart_housing_pressure_quarterly;

-- 3b) Drilldown: which rows have missing CPI (informational; used to document CPI coverage gap)
SELECT
  boligtype_code,
  quarter,
  quarter_start,
  hpi_index,
  cpi_index,
  cpi_method
FROM mart_housing_pressure_quarterly
WHERE cpi_index IS NULL
ORDER BY quarter_start;

-- 4) Coverage by boligtype_code (n_quarters should equal quarter_count for each boligtype present)
SELECT
  boligtype_code,
  COUNT(*) AS n_quarters,
  MIN(quarter_start) AS min_quarter_start,
  MAX(quarter_start) AS max_quarter_start
FROM mart_housing_pressure_quarterly
GROUP BY 1
ORDER BY boligtype_code;

-- 5) Sanity: index values should be > 0 (expect 0 nonpositive rows)
SELECT
  SUM(CASE WHEN hpi_index <= 0 THEN 1 ELSE 0 END) AS nonpositive_hpi_index_rows,
  SUM(CASE WHEN cpi_index <= 0 THEN 1 ELSE 0 END) AS nonpositive_cpi_index_rows
FROM mart_housing_pressure_quarterly;

-- 6) KPI comparison vs Part 2 reference columns (expect max diffs ~0 for HPI + proxy)
SELECT
  MAX(ABS(hpi_qoq_pct - hpi_index_qoq_pct_part2)) AS max_abs_diff_hpi_qoq_pct,
  MAX(ABS(hpi_yoy_pct - hpi_index_yoy_pct_part2)) AS max_abs_diff_hpi_yoy_pct,
  MAX(ABS(real_hpi_proxy - real_hpi_proxy_part2)) AS max_abs_diff_real_hpi_proxy
FROM mart_housing_pressure_quarterly
WHERE hpi_qoq_pct IS NOT NULL
  AND hpi_index_qoq_pct_part2 IS NOT NULL;

-- 7) SA availability (informational - many NULLs are expected if SA not provided early)
SELECT
  SUM(CASE WHEN hpi_sa_index IS NULL THEN 1 ELSE 0 END) AS hpi_sa_index_null_rows,
  SUM(CASE WHEN hpi_sa_qoq_pct IS NULL THEN 1 ELSE 0 END) AS hpi_sa_qoq_pct_null_rows,
  SUM(CASE WHEN hpi_sa_yoy_pct IS NULL THEN 1 ELSE 0 END) AS hpi_sa_yoy_pct_null_rows
FROM mart_housing_pressure_quarterly;
