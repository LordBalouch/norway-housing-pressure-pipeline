-- 03_eda_queries.sql
-- Purpose: Analyst-style EDA queries over the quarterly mart.
-- Note: Current mart has boligtype_count = 1 (boligtype_code = 0).
-- CPI is missing for 2025Q1–2025Q3, so real proxy metrics will be NULL there.

-- Q1) What is the latest available quarter and its headline KPIs
SELECT
  quarter,
  quarter_start,
  hpi_index,
  hpi_qoq_pct,
  hpi_yoy_pct,
  policy_rate_end,
  policy_rate_avg,
  cpi_index,
  real_hpi_proxy
FROM mart_housing_pressure_quarterly
ORDER BY quarter_start DESC
LIMIT 1;

-- Q2) Top 5 biggest QoQ increases (largest positive moves)
SELECT
  quarter,
  quarter_start,
  hpi_index,
  hpi_qoq_pct,
  policy_rate_end,
  policy_delta_sum
FROM mart_housing_pressure_quarterly
WHERE hpi_qoq_pct IS NOT NULL
ORDER BY hpi_qoq_pct DESC
LIMIT 5;

-- Q3) Top 5 biggest QoQ decreases (largest negative moves)
SELECT
  quarter,
  quarter_start,
  hpi_index,
  hpi_qoq_pct,
  policy_rate_end,
  policy_delta_sum
FROM mart_housing_pressure_quarterly
WHERE hpi_qoq_pct IS NOT NULL
ORDER BY hpi_qoq_pct ASC
LIMIT 5;

-- Q4) Top 5 strongest YoY growth quarters
SELECT
  quarter,
  quarter_start,
  hpi_index,
  hpi_yoy_pct,
  policy_rate_end,
  cpi_index
FROM mart_housing_pressure_quarterly
WHERE hpi_yoy_pct IS NOT NULL
ORDER BY hpi_yoy_pct DESC
LIMIT 5;

-- Q5) Top 5 weakest YoY growth quarters (most negative / lowest)
SELECT
  quarter,
  quarter_start,
  hpi_index,
  hpi_yoy_pct,
  policy_rate_end,
  cpi_index
FROM mart_housing_pressure_quarterly
WHERE hpi_yoy_pct IS NOT NULL
ORDER BY hpi_yoy_pct ASC
LIMIT 5;

-- Q6) Which periods show the highest short-run volatility
-- Rolling 8-quarter standard deviation of QoQ change
WITH vol AS (
  SELECT
    quarter,
    quarter_start,
    hpi_qoq_pct,
    STDDEV_SAMP(hpi_qoq_pct) OVER (
      ORDER BY quarter_start
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS qoq_std_8q
  FROM mart_housing_pressure_quarterly
  WHERE hpi_qoq_pct IS NOT NULL
)
SELECT
  quarter,
  quarter_start,
  hpi_qoq_pct,
  qoq_std_8q
FROM vol
WHERE qoq_std_8q IS NOT NULL
ORDER BY qoq_std_8q DESC
LIMIT 10;

-- Q7) How often do rate hikes coincide with negative QoQ house price change
-- Define a "tightening quarter" as policy_rate_end increasing vs previous quarter
WITH t AS (
  SELECT
    quarter,
    quarter_start,
    hpi_qoq_pct,
    policy_rate_end,
    policy_rate_end - LAG(policy_rate_end) OVER (ORDER BY quarter_start) AS policy_end_delta
  FROM mart_housing_pressure_quarterly
)
SELECT
  COUNT(*) AS n_quarters_with_policy_delta,
  SUM(CASE WHEN policy_end_delta > 0 THEN 1 ELSE 0 END) AS n_tightening_quarters,
  SUM(CASE WHEN policy_end_delta > 0 AND hpi_qoq_pct < 0 THEN 1 ELSE 0 END) AS n_tightening_and_negative_qoq,
  ROUND(
    100.0 * SUM(CASE WHEN policy_end_delta > 0 AND hpi_qoq_pct < 0 THEN 1 ELSE 0 END)
    / NULLIF(SUM(CASE WHEN policy_end_delta > 0 THEN 1 ELSE 0 END), 0),
    2
  ) AS pct_tightening_quarters_with_negative_qoq
FROM t
WHERE policy_end_delta IS NOT NULL
  AND hpi_qoq_pct IS NOT NULL;

-- Q8) Simple correlation check: policy_rate_end vs YoY growth (informational, not causal)
SELECT
  CORR(policy_rate_end, hpi_yoy_pct) AS corr_policy_end_vs_hpi_yoy
FROM mart_housing_pressure_quarterly
WHERE policy_rate_end IS NOT NULL
  AND hpi_yoy_pct IS NOT NULL;

-- Q9) Data caveat query: which quarters have missing CPI
SELECT
  quarter,
  quarter_start,
  cpi_index,
  real_hpi_proxy,
  real_hpi_proxy_yoy_pct
FROM mart_housing_pressure_quarterly
WHERE cpi_index IS NULL
ORDER BY quarter_start;
