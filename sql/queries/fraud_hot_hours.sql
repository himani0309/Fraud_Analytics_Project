-- Fraud rate by hour of day (0..23)
WITH base AS (
  SELECT
    FLOOR(MOD("Time", 86400) / 3600.0)::INT AS hour,
    "Class"::INT AS class
  FROM transactions
)
SELECT
  hour,
  COUNT(*) AS total_txn,
  SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_txn,
  ROUND(SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*), 6) AS fraud_rate
FROM base
GROUP BY hour
ORDER BY fraud_rate DESC, total_txn DESC;
