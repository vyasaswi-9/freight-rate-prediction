"""Cleaning for the freight-rate data.

One Cleaner is fit on TRAINING data only, then applied unchanged to train,
validation, and the December chart file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHT_CAP = 47_500.0     # clipping seen in the EDA (about 30x neighbouring bins)
WEIGHT_FLOOR = 5_000.0    # smaller pile-up at the low end
DIST_BINS = [0, 300, 600, 1000, 1500, 2000, 2500, np.inf]
OUTLIER_LOW, OUTLIER_HIGH = 0.6, 1.5   # DECISION: gap in rpm_ratio sits at 0.5-0.8 and 1.3-2.2


class Cleaner:
    """fit() learns numbers from training data; transform() applies them."""

    def fit(self, train: pd.DataFrame) -> "Cleaner":
        w = train["weight"].abs()
        self.weight_median_ = w.median()
        self.weight_median_by_equip_ = w.groupby(train["equipment"]).median().to_dict()
        if "market_index" in train.columns:
            self.market_index_median_ = train["market_index"].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="raise")

        # --- weight ---
        # DECISION: negatives are sign flips (their |weight| distribution matches the positives)
        out["weight_was_negative"] = (out["weight"] < 0).astype(int)
        out["weight"] = out["weight"].abs()

        # DECISION: keep capped/floored values, add flags instead of dropping or nulling
        out["weight_capped"] = (out["weight"] == WEIGHT_CAP).astype(int)
        out["weight_floored"] = (out["weight"] == WEIGHT_FLOOR).astype(int)

        # DECISION: median by equipment, learned from train only, plus a missing flag
        out["weight_missing"] = out["weight"].isna().astype(int)
        fill = out["equipment"].map(self.weight_median_by_equip_).fillna(self.weight_median_)
        out["weight"] = out["weight"].fillna(fill)

        # --- market_index (only present in train and validation, not the December file) ---
        if "market_index" in out.columns:
            out["market_index_missing"] = out["market_index"].isna().astype(int)
            # DECISION: market_index moves by date, so fill with that date's average across
            # loads in the same file (features only, no target involved). Fall back to the
            # train median for dates where every value is missing.
            by_date = out.groupby("date")["market_index"].transform("mean")
            out["market_index"] = out["market_index"].fillna(by_date).fillna(self.market_index_median_)

        # distance: DECISION: leave unchanged. The 70-mile floor is real; rates follow it.
        return out


# ---------- target-based outlier handling (TRAINING data only) ----------

def fit_rate_baseline(train: pd.DataFrame, target: str = "posted_rate") -> pd.Series:
    """Median rate per mile by equipment and distance band, from the given rows only."""
    d = train.assign(
        rpm=train[target] / train["distance"],
        dist_bin=pd.cut(train["distance"], DIST_BINS),
    )
    return d.groupby(["equipment", "dist_bin"], observed=True)["rpm"].median()


def outlier_mask(df: pd.DataFrame, baseline: pd.Series, target: str = "posted_rate") -> pd.Series:
    """True for rows whose rate per mile is far from the baseline. Needs the target,
    so it can only ever be used on labelled data, never on validation."""
    d = df.assign(dist_bin=pd.cut(df["distance"], DIST_BINS))
    keys = pd.MultiIndex.from_frame(d[["equipment", "dist_bin"]])
    expected = pd.Series(baseline.reindex(keys).to_numpy(), index=df.index)
    ratio = (df[target] / df["distance"]) / expected
    return (ratio < OUTLIER_LOW) | (ratio > OUTLIER_HIGH)