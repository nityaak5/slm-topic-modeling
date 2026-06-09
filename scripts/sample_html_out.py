"""
Create a 200-row stratified random sample of html_out_stance.csv,
stratified by country × domain using proportional allocation
with the largest-remainder method (min 1 per stratum).
"""

import numpy as np
import pandas as pd

SRC = "data_in/html_out/html_out_stance.csv"
OUT = "data_in/html_out/html_out_stance_sample100.csv"
TARGET = 100
SEED = 42


def allocate(strata: pd.DataFrame, target: int) -> pd.DataFrame:
    """Proportional allocation with largest-remainder, min 1 per stratum."""
    total = strata["count"].sum()
    strata = strata.copy()
    strata["exact"] = strata["count"] / total * target
    strata["alloc"] = strata["exact"].apply(np.floor).astype(int)
    # Guarantee at least 1, but never exceed available rows
    strata["alloc"] = strata.apply(
        lambda r: max(1, min(r["alloc"], r["count"])), axis=1
    )
    # Distribute remainder to strata with largest fractional parts
    strata["remainder"] = strata["exact"] - strata["alloc"]
    strata["capacity"] = strata["count"] - strata["alloc"]
    remaining = target - strata["alloc"].sum()
    strata = strata.sort_values("remainder", ascending=False).reset_index(drop=True)
    for i in strata.index:
        if remaining <= 0:
            break
        if strata.at[i, "capacity"] > 0:
            strata.at[i, "alloc"] += 1
            remaining -= 1
    return strata


def main():
    df = pd.read_csv(SRC)
    strata = df.groupby(["country", "domain"]).size().reset_index(name="count")
    strata = allocate(strata, TARGET)

    rng = np.random.default_rng(SEED)
    parts = []
    for _, row in strata.iterrows():
        pool = df[(df["country"] == row["country"]) & (df["domain"] == row["domain"])]
        n = int(row["alloc"])
        sampled = pool.sample(n=n, random_state=int(rng.integers(1e6)))
        parts.append(sampled)

    sample = pd.concat(parts).sort_values("id").reset_index(drop=True)
    sample["id"] = range(1, len(sample) + 1)

    sample.to_csv(OUT, index=False)
    print(f"Saved {len(sample)} rows → {OUT}")
    print()
    print("country distribution:")
    print(sample["country"].value_counts().to_string())
    print()
    print("domain distribution:")
    print(sample["domain"].value_counts().to_string())


if __name__ == "__main__":
    main()
