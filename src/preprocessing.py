"""
preprocessing.py
-----------------
Handles data loading, validation, and chronological train/val/test splitting.

DATA LEAKAGE PREVENTION — DOCUMENTED
======================================
For any target email at timestamp T:

  ALLOWED  : records where timestamp < T  (historical memory pool)
  FORBIDDEN: target email's own user_response
             any record with timestamp >= T (future records)
             user_id as a direct predictive feature

The train/val/test split is performed CHRONOLOGICALLY (no shuffling).
The split boundary is computed on sorted timestamps so:
  - All training records precede all validation records.
  - All validation records precede all test records.
  - This exactly mirrors the real-world deployment scenario.

No val/test text statistics are used when fitting any transformer (TF-IDF).
"""

import pathlib
import pandas as pd
from typing import Tuple

# -- Constants -----------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
RAW_CSV = ROOT / "data" / "raw" / "emails.csv"

REQUIRED_COLUMNS = [
    "record_id", "user_id", "timestamp",
    "email_subject", "incoming_email", "user_response", "decision",
]

VALID_DECISIONS = {"ACCEPT", "REJECT", "MAYBE", "DEFER", "REQUEST_MORE_INFORMATION"}

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
# TEST_RATIO  = 0.20  (implicit: remainder)


def load_raw(path: pathlib.Path = RAW_CSV) -> pd.DataFrame:
    """Load and minimally validate the raw CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Run:  python scripts/generate_dataset.py"
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Schema check
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Decision validity check
    invalid = set(df["decision"].unique()) - VALID_DECISIONS
    if invalid:
        raise ValueError(f"Unknown decision labels found: {invalid}")

    # Sort chronologically (critical for leakage prevention)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"[preprocessing] Loaded {len(df)} records from {path.name}")
    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into train / validation / test sets
    using strictly chronological ordering.

    WHY NOT RANDOM SPLIT?
    A random split would allow the model to train on future emails and
    evaluate on past emails — a severe form of data leakage that would
    produce optimistically biased metrics.  In real deployment the model
    only has access to emails that arrived before the target email.

    Returns:
        (train_df, val_df, test_df) — non-overlapping, time-ordered slices.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    # Sanity: all train timestamps < all val timestamps < all test timestamps
    assert train_df["timestamp"].max() <= val_df["timestamp"].min(), \
        "Leakage detected: train/val timestamp overlap!"
    assert val_df["timestamp"].max() <= test_df["timestamp"].min(), \
        "Leakage detected: val/test timestamp overlap!"

    print(
        f"[preprocessing] Chronological split → "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df


def describe_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Print a human-readable summary of the three splits."""
    for name, df in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        dist = df["decision"].value_counts().to_dict()
        print(
            f"  {name:5s} | n={len(df):4d} | "
            f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()} | "
            f"decisions: {dist}"
        )
