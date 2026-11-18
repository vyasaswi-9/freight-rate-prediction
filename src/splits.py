"""Time-based expanding-window folds, plus per-fold cleaning with no leakage."""
from __future__ import annotations

import pandas as pd

from clean import Cleaner, fit_rate_baseline, outlier_mask

# DECISION: expanding window, 2-month test windows, 3 folds.
# Each tuple is (last train date, test start, test end), all inclusive.
FOLDS = [
    ("2025-04-30", "2025-05-01", "2025-06-30"),
    ("2025-06-30", "2025-07-01", "2025-08-31"),
    ("2025-08-31", "2025-09-01", "2025-10-31"),   # closest to the real Nov-Dec task
]


def make_folds(df: pd.DataFrame, folds=FOLDS):
    """Yield (fold_number, train_idx, test_idx) using the raw 'date' column."""
    dates = pd.to_datetime(df["date"])
    for i, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train_idx = df.index[dates <= pd.Timestamp(train_end)]
        test_idx = df.index[(dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))]
        assert dates[train_idx].max() < dates[test_idx].min(), "train/test overlap in time"
        yield i, train_idx, test_idx


def prepare_fold(raw: pd.DataFrame, train_idx, test_idx, target: str = "posted_rate"):
    """Fit the Cleaner and the outlier baseline on the fold's TRAIN rows only,
    then apply to both parts. Returns cleaned train and test frames.

    is_outlier is computed for the test rows too, but only so metrics can be
    reported with and without them. It is never used to train anything on test rows."""
    train_raw, test_raw = raw.loc[train_idx], raw.loc[test_idx]

    cleaner = Cleaner().fit(train_raw)
    train, test = cleaner.transform(train_raw), cleaner.transform(test_raw)

    baseline = fit_rate_baseline(train, target)
    train["is_outlier"] = outlier_mask(train, baseline, target)
    test["is_outlier"] = outlier_mask(test, baseline, target)
    return train, test, cleaner, baseline


def unseen_report(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Share of test rows whose cities or lane never appear in the fold's train rows."""
    lane = lambda d: d["pickup"] + " -> " + d["delivery"]
    return {
        "unseen_pickup": (~test["pickup"].isin(set(train["pickup"]))).mean(),
        "unseen_delivery": (~test["delivery"].isin(set(train["delivery"]))).mean(),
        "unseen_lane": (~lane(test).isin(set(lane(train)))).mean(),
    }