-- Fraud rate by transaction amount buckets
WITH b AS (
  SELECT
    CASE
      WHEN "Amount" < 10   THEN '[0,10)'
      WHEN "Amount" < 50   THEN '[10,50)'
      WHEN "Amount" < 100  THEN '[50,100)'
      WHEN "Amount" < 500  THEN '[100,500)'
      WHEN "Amount" < 1000 THEN '[500,1000)'
      ELSE '1000+'
    END AS amt_bucket,
    "Class"::INT AS class
  FROM transactions
)
SELECT
  amt_bucket,
  COUNT(*) AS total_txn,
  SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_txn,
  ROUND(SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*), 6) AS fraud_rate
FROM b
GROUP BY amt_bucket
ORDER BY
  CASE amt_bucket
    WHEN '[0,10)' THEN 1 WHEN '[10,50)' THEN 2 WHEN '[50,100)' THEN 3
    WHEN '[100,500)' THEN 4 WHEN '[500,1000)' THEN 5 ELSE 6
  END;
