from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier

PROC = Path("data/processed/cc_processed.csv")

def pick_threshold(y_true, proba, min_precision=0.90):
    """Return threshold that gives highest recall with precision >= min_precision."""
    pr, rc, thr = precision_recall_curve(y_true, proba)
    ok = np.where(pr >= min_precision)[0]
    if len(ok) == 0:
        return 0.5  # fallback
    i = ok[np.argmax(rc[ok])]
    # precision_recall_curve returns thr len = n-1 vs pr/rc len = n
    return float(thr[i-1]) if i > 0 and i-1 < len(thr) else 0.5

def main():
    if not PROC.exists():
        raise FileNotFoundError(f"Run feature_engineering.py first; not found: {PROC.resolve()}")

    df = pd.read_csv(PROC)
    y = df["Class"].values
    X = df.drop(columns=["Class"]).values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Class imbalance handling for XGB via scale_pos_weight
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    spw = float(neg / max(pos, 1))

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        n_jobs=-1,
        eval_metric="auc",
        tree_method="hist"
    )
    xgb.fit(X_tr, y_tr)

    proba = xgb.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)

    # choose threshold to keep precision >= 0.90 (adjust as needed)
    thr = pick_threshold(y_te, proba, min_precision=0.90)
    pred = (proba >= thr).astype(int)

    print(f"ROC-AUC: {auc:.4f} | threshold: {thr:.4f}")
    print("\nClassification report (precision>=0.90 target):\n",
          classification_report(y_te, pred, digits=4))

if __name__ == "__main__":
    main()
