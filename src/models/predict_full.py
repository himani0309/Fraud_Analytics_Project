import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

from xgboost import XGBClassifier


# ------------------------- config -------------------------
SEED = 42
PROCESSED_PATH = Path("data/processed/cc_processed.csv")
REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "Class"
MIN_PRECISION = 0.90
# ----------------------------------------------------------


def pick_threshold(y_true, proba, min_precision=0.90):
    """Return threshold achieving precision >= min_precision with max recall.
       If not achievable, return threshold at best precision."""
    pr, rc, thr = precision_recall_curve(y_true, proba)
    ok = np.where(pr >= min_precision)[0]
    if len(ok) == 0:
        # fallback—best precision available
        j = int(np.argmax(pr))
        t = thr[j - 1] if j > 0 else 0.5
        return float(t), float(pr[j]), float(rc[j]), False
    i = ok[np.argmax(rc[ok])]
    t = thr[i - 1] if i > 0 else 0.5
    return float(t), float(pr[i]), float(rc[i]), True


def save_confusion_matrix(y_true, y_pred, title, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_roc_curve(y_true, proba, title, outpath):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main():
    # 1) Load
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Processed CSV not found at {PROCESSED_PATH}. "
                                "Run: python src/data/feature_engineering.py")
    df = pd.read_csv(PROCESSED_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Column '{TARGET}' not found in {PROCESSED_PATH}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # 2) Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # 3) Model (class imbalance handled via scale_pos_weight)
    pos = int(y_tr.sum())
    neg = int((y_tr == 0).sum())
    spw = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="auc",
        n_jobs=-1,
        random_state=SEED,
        scale_pos_weight=spw
    )
    model.fit(X_tr, y_tr)

    # 4) Scores & threshold
    proba = model.predict_proba(X_te)[:, 1]
    roc_auc = roc_auc_score(y_te, proba)

    thr, p_at_t, r_at_t, met_target = pick_threshold(y_te, proba, MIN_PRECISION)
    y_hat = (proba >= thr).astype(int)

    # 5) Save predictions
    out_all = REPORTS_DIR / "predictions.csv"
    out_top = REPORTS_DIR / "top_100_high_risk.csv"

    te_df = X_te.copy()
    te_df["true_class"] = y_te.values
    te_df["proba_fraud"] = proba
    te_df["pred_class"] = y_hat
    te_sorted = te_df.sort_values("proba_fraud", ascending=False)
    te_sorted.to_csv(out_all, index=False)
    te_sorted.head(100).to_csv(out_top, index=False)

    # 6) Figures
    save_confusion_matrix(y_te, y_hat, "Confusion Matrix - XGBoost (tuned)", FIG_DIR / "04_cm_xgboost.png")
    save_roc_curve(y_te, proba, "ROC Curve - XGBoost", FIG_DIR / "05_roc_xgboost.png")

    # 7) Console summary
    status = "✅ met" if met_target else "⚠️ not met (picked best available)"
    print("\n=== PREDICTION EXPORT SUMMARY ===")
    print(f"Test size: {len(y_te)} | Positives (fraud): {int(y_te.sum())}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"Threshold selected : {thr:.4f} | Precision: {p_at_t:.4f} | Recall: {r_at_t:.4f} -> target {status}")
    print(f"Saved: {out_all} and {out_top}")
    print(f"Figures: {FIG_DIR / '04_cm_xgboost.png'} , {FIG_DIR / '05_roc_xgboost.png'}")


if __name__ == "__main__":
    main()
