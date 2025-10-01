-- Fraud rate per day (0 = first day in dataset)
WITH base AS (
  SELECT
    FLOOR("Time" / 86400.0)::INT AS day_idx,
    "Class"::INT AS class
  FROM transactions
)
SELECT
  day_idx,
  COUNT(*) AS total_txn,
  SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_txn,
  ROUND(SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*), 6) AS fraud_rate
FROM base
GROUP BY day_idx
ORDER BY day_idx;
