from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RAW = Path("data/raw/creditcard.csv")
FIGDIR = Path("reports/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW.resolve()}")

    df = pd.read_csv(RAW)
    print("Shape:", df.shape)
    print("\nClass counts:\n", df["Class"].value_counts())
    print("Fraud %:", round(100 * df["Class"].mean(), 4))

    # 1) Class balance
    ax = df["Class"].value_counts().rename({0: "Non-Fraud", 1: "Fraud"}).plot(
        kind="bar", title="Class Balance"
    )
    ax.set_ylabel("Count")
    plt.tight_layout(); plt.savefig(FIGDIR / "01_class_balance.png"); plt.close()

    # 2) Amount distribution (fraud vs non-fraud)
    plt.figure()
    df.loc[df.Class == 0, "Amount"].plot(kind="hist", bins=60, alpha=0.6, label="Non-Fraud")
    df.loc[df.Class == 1, "Amount"].plot(kind="hist", bins=60, alpha=0.6, label="Fraud")
    plt.legend(); plt.title("Amount Distribution"); plt.xlabel("Amount"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(FIGDIR / "02_amount_distribution.png"); plt.close()

    # 3) Fraud rate by hour
    if "Time" in df.columns:
        df["hour"] = (df["Time"] % (24 * 3600)) // 3600
        ax = df.groupby("hour")["Class"].mean().plot(title="Fraud Rate by Hour")
        ax.set_ylabel("Rate")
        plt.tight_layout(); plt.savefig(FIGDIR / "03_fraud_rate_by_hour.png"); plt.close()

    print("Saved figures to:", FIGDIR.resolve())

if __name__ == "__main__":
    main()
