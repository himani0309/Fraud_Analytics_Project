import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RAW = Path("data/raw/creditcard.csv")
OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading dataset...")
    df = pd.read_csv(RAW)
    print(f"Shape: {df.shape}")
    print(df['Class'].value_counts())
    print("Fraud %:", 100 * df['Class'].mean())

    # 1. Class Balance (Bar + Pie)
    counts = df['Class'].value_counts().rename({0:'Non-Fraud',1:'Fraud'})

    plt.figure()
    counts.plot(kind='bar', log=True)
    plt.title("Class Balance (log scale)")
    plt.ylabel("Count (log)")
    plt.savefig(OUT / "01_class_balance_log.png", dpi=160)
    plt.close()

    plt.figure()
    counts.plot.pie(autopct="%.3f%%", labels=['Non-Fraud','Fraud'])
    plt.title("Fraud vs Non-Fraud Distribution")
    plt.ylabel("")
    plt.savefig(OUT / "01_class_balance_pie.png", dpi=160)
    plt.close()

    # 2. Amount Distribution (log transform)
    df['log_amount'] = np.log1p(df['Amount'])

    plt.figure()
    df[df['Class']==0]['log_amount'].plot(kind='hist', bins=50, alpha=0.6, label='Non-Fraud')
    df[df['Class']==1]['log_amount'].plot(kind='hist', bins=50, alpha=0.6, label='Fraud')
    plt.legend()
    plt.title("Log-Transformed Amount Distribution")
    plt.xlabel("log(Amount+1)")
    plt.ylabel("Count")
    plt.savefig(OUT / "02_amount_distribution_log.png", dpi=160)
    plt.close()

    # 3. Fraud Rate by Hour
    df['hour'] = (df['Time'] % (24*3600)) // 3600
    hr = df.groupby('hour')['Class'].mean()

    plt.figure()
    hr.plot()
    plt.title("Fraud Rate by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Fraud Rate")
    plt.savefig(OUT / "03_fraud_rate_by_hour.png", dpi=160)
    plt.close()

    print(f"Saved updated figures to {OUT.resolve()}")

if __name__ == "__main__":
    main()
