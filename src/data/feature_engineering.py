from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/creditcard.csv")
OUT = Path("data/processed/cc_processed.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW.resolve()}")

    df = pd.read_csv(RAW)

    # Basic time & amount features
    if "Time" in df.columns:
        df["hour"] = (df["Time"] % (24 * 3600)) // 3600
        df["is_night"] = df["hour"].isin([0,1,2,3,4]).astype(int)

    if "Amount" in df.columns:
        df["log_amount"] = np.log1p(df["Amount"])

    # NOTE: V1..V28 are anonymized PCA components; keep them as-is.
    df.to_csv(OUT, index=False)
    print(f"Saved processed dataset → {OUT.resolve()}  | shape: {df.shape}")

if __name__ == "__main__":
    main()
