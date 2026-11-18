"""Gradient boosting on log(rate per mile). Same fit/predict interface as the baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import build_X

TARGET = "posted_rate"


class GBMModel:
    def __init__(self, groups: list[str], **params):
        self.groups = groups
        # DECISION: modest defaults; the variants compared in 06_models.ipynb are passed in as params
        self.params = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                           min_samples_leaf=50, random_state=0) | params

    def fit(self, train: pd.DataFrame):
        X = build_X(train, self.groups)                    # build features on ALL rows first
        keep = ~train["is_outlier"]                        # then drop flagged outliers for fitting
        y = np.log(train[TARGET] / train["distance"])[keep]
        self.model_ = HistGradientBoostingRegressor(**self.params).fit(X[keep], y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        log_rpm = self.model_.predict(build_X(df, self.groups))
        return pd.Series(df["distance"].to_numpy() * np.exp(log_rpm), index=df.index)