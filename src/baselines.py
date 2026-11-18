"""A ladder of simple baselines. Each has fit(train_df) and predict(df)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from clean import DIST_BINS

TARGET = "posted_rate"
MIN_GROUP = 5   # DECISION: ignore groups with fewer rows than this (medians of 1-2 rows are noise)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["dist_bin"] = pd.cut(d["distance"], DIST_BINS)
    d["lane"] = d["pickup"] + " -> " + d["delivery"]
    return d


def _index(d: pd.DataFrame, keys: list[str]):
    return pd.Index(d[keys[0]]) if len(keys) == 1 else pd.MultiIndex.from_frame(d[keys])


class RpmBaseline:
    """Predict distance x median rate-per-mile. 'levels' is a list of key sets,
    tried from most specific to least; anything not found falls through to the
    next level, and finally to the global median."""

    def __init__(self, levels: list[list[str]]):
        self.levels = levels

    def fit(self, train: pd.DataFrame):
        d = _prep(train[~train["is_outlier"]])          # fit on non-outlier rows only
        d["rpm"] = d[TARGET] / d["distance"]
        self.global_ = d["rpm"].median()
        self.tables_ = []
        for keys in self.levels:
            g = d.groupby(keys, observed=True)["rpm"].agg(["median", "count"])
            self.tables_.append(g.loc[g["count"] >= MIN_GROUP, "median"])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        d = _prep(df)
        rpm = pd.Series(np.nan, index=d.index)
        for keys, table in zip(self.levels, self.tables_):
            looked_up = pd.Series(table.reindex(_index(d, keys)).to_numpy(), index=d.index)
            rpm = rpm.fillna(looked_up)
        return d["distance"] * rpm.fillna(self.global_)


class LinearBaseline:
    """Plain linear regression on distance, weight and equipment dummies."""

    def _X(self, df):
        return np.column_stack([df["distance"], df["weight"],
                                (df["equipment"] == "Reefer"), (df["equipment"] == "Flatbed")]).astype(float)

    def fit(self, train):
        fit_rows = train[~train["is_outlier"]]
        self.model_ = LinearRegression().fit(self._X(fit_rows), fit_rows[TARGET])
        return self

    def predict(self, df):
        return pd.Series(self.model_.predict(self._X(df)), index=df.index)


BASELINES = {
    "1 global rate/mile":       lambda: RpmBaseline([]),
    "2 by equipment":           lambda: RpmBaseline([["equipment"]]),
    "3 equipment x dist band":  lambda: RpmBaseline([["equipment", "dist_bin"]]),
    "4 lane x equipment":       lambda: RpmBaseline([["lane", "equipment"], ["equipment", "dist_bin"]]),
    "5 linear regression":      lambda: LinearBaseline(),
}