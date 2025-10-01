from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

PROC = Path("data/processed/cc_processed.csv")

def main():
    if not PROC.exists():
        raise FileNotFoundError(f"Run feature_engineering.py first; not found: {PROC.resolve()}")

    df = pd.read_csv(PROC)
    y = df["Class"]
    X = df.drop(columns=["Class"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle heavy imbalance via SMOTE
    sm = SMOTE(random_state=42)
    X_trb, y_trb = sm.fit_resample(X_tr, y_tr)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_trb, y_trb)

    proba = clf.predict_proba(X_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    pr, rc, _ = precision_recall_curve(y_te, proba)
    pr_auc = auc(rc, pr)

    print("ROC-AUC:", round(roc_auc_score(y_te, proba), 4))
    print("PR-AUC :", round(pr_auc, 4))
    print("\nClassification report (threshold=0.5):\n",
          classification_report(y_te, pred, digits=4))

if __name__ == "__main__":
    main()
