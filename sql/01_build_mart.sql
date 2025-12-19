-- 01_build_mart.sql
-- Build a quarterly mart (one row per boligtype_code + quarter) with KPIs computed in SQL.
-- Base input: data/processed/analysis_ready_quarterly.csv (Part 2 output)
-- CPI CAVEAT: CPI is an annual value repeated each quarter (documented in Part 2).
-- Therefore, "real_*" measures are proxies and must be interpreted carefully.

CREATE OR REPLACE VIEW v_analysis_ready_quarterly AS
SELECT
  CAST(boligtype_code AS INTEGER)                 AS boligtype_code,
  CAST(quarter AS VARCHAR)                        AS quarter,
  TRY_CAST(quarter_start AS DATE)                 AS quarter_start,

  CAST(hpi_index AS DOUBLE)                       AS hpi_index,
  CAST(hpi_sa_index AS DOUBLE)                    AS hpi_sa_index,

  CAST(cpi_index AS DOUBLE)                       AS cpi_index,
  CAST(cpi_method AS VARCHAR)                     AS cpi_method,

  CAST(policy_rate_avg AS DOUBLE)                 AS policy_rate_avg,
  CAST(policy_rate_end AS DOUBLE)                 AS policy_rate_end,
  CAST(policy_rate_min AS DOUBLE)                 AS policy_rate_min,
  CAST(policy_rate_max AS DOUBLE)                 AS policy_rate_max,
  CAST(policy_obs_count AS BIGINT)                AS policy_obs_count,

  CAST(policy_change_events AS DOUBLE)            AS policy_change_events,
  CAST(policy_delta_sum AS DOUBLE)                AS policy_delta_sum,
  CAST(policy_delta_abs_sum AS DOUBLE)            AS policy_delta_abs_sum,

  -- Part 2 reference KPIs (kept for validation vs SQL recomputation)
  CAST(hpi_index_qoq_pct AS DOUBLE)               AS hpi_index_qoq_pct_part2,
  CAST(hpi_index_yoy_pct AS DOUBLE)               AS hpi_index_yoy_pct_part2,
  CAST(hpi_sa_index_qoq_pct AS DOUBLE)            AS hpi_sa_index_qoq_pct_part2,
  CAST(hpi_sa_index_yoy_pct AS DOUBLE)            AS hpi_sa_index_yoy_pct_part2,
  CAST(cpi_yoy_pct AS DOUBLE)                     AS cpi_yoy_pct_part2,
  CAST(real_hpi_proxy AS DOUBLE)                  AS real_hpi_proxy_part2,
  CAST(real_hpi_sa_proxy AS DOUBLE)               AS real_hpi_sa_proxy_part2

FROM read_csv_auto('data/processed/analysis_ready_quarterly.csv', header = TRUE);

CREATE OR REPLACE TABLE mart_housing_pressure_quarterly AS
WITH base AS (
  SELECT *
  FROM v_analysis_ready_quarterly
)
SELECT
  boligtype_code,
  quarter,
  quarter_start,

  hpi_index,
  hpi_sa_index,
  cpi_index,
  cpi_method,

  policy_rate_avg,
  policy_rate_end,
  policy_rate_min,
  policy_rate_max,
  policy_obs_count,
  policy_change_events,
  policy_delta_sum,
  policy_delta_abs_sum,

  -- Part 2 KPI references
  hpi_index_qoq_pct_part2,
  hpi_index_yoy_pct_part2,
  hpi_sa_index_qoq_pct_part2,
  hpi_sa_index_yoy_pct_part2,
  cpi_yoy_pct_part2,
  real_hpi_proxy_part2,
  real_hpi_sa_proxy_part2,

  -- SQL KPI recomputation (percent units, e.g., 0.53 means 0.53%)
  100 * (hpi_index / NULLIF(LAG(hpi_index) OVER (PARTITION BY boligtype_code ORDER BY quarter_start), 0) - 1)
    AS hpi_qoq_pct,

  100 * (hpi_index / NULLIF(LAG(hpi_index, 4) OVER (PARTITION BY boligtype_code ORDER BY quarter_start), 0) - 1)
    AS hpi_yoy_pct,

  100 * (hpi_sa_index / NULLIF(LAG(hpi_sa_index) OVER (PARTITION BY boligtype_code ORDER BY quarter_start), 0) - 1)
    AS hpi_sa_qoq_pct,

  100 * (hpi_sa_index / NULLIF(LAG(hpi_sa_index, 4) OVER (PARTITION BY boligtype_code ORDER BY quarter_start), 0) - 1)
    AS hpi_sa_yoy_pct,

  -- "Real" proxy (CPI caveat applies)
  (hpi_index / NULLIF(cpi_index, 0)) AS real_hpi_proxy,
  (hpi_sa_index / NULLIF(cpi_index, 0)) AS real_hpi_sa_proxy,

  -- Optional: YoY change of proxy (still subject to CPI caveat)
  100 * (
    (hpi_index / NULLIF(cpi_index, 0))
    / NULLIF(LAG(hpi_index / NULLIF(cpi_index, 0), 4) OVER (PARTITION BY boligtype_code ORDER BY quarter_start), 0)
    - 1
  ) AS real_hpi_proxy_yoy_pct

FROM base
ORDER BY boligtype_code, quarter_start;

-- Quick summary for the runner output
SELECT
  COUNT(*) AS row_count,
  COUNT(DISTINCT boligtype_code) AS boligtype_count,
  MIN(quarter_start) AS min_quarter_start,
  MAX(quarter_start) AS max_quarter_start
FROM mart_housing_pressure_quarterly;
