"""Prepare sample dataset for DVC pipeline (toy example)."""
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    df.to_csv(out_dir / "sample_prepared.csv", index=False)


if __name__ == "__main__":
    main()
