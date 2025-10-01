-- Top 100 highest-risk transactions (by predicted probability)
SELECT
  txn_id,
  proba_fraud,
  pred_class,
  true_class
FROM predictions
ORDER BY proba_fraud DESC
LIMIT 100;
